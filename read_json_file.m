function data = read_json_file(filename)
    % readJSONFile reads a JSON file and returns the data as a MATLAB structure.
    %
    % Usage:
    %   data = readJSONFile('filename.json')
    %
    % Input:
    %   - filename: string, the name of the JSON file to read.
    %
    % Output:
    %   - data: MATLAB structure, the data contained in the JSON file.

    % Check if the file exists
    if ~isfile(filename)
        error('File %s does not exist.', filename);
    end

    % Open the file
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file %s.', filename);
    end

    % Read the file content
    raw = fread(fid, inf);
    str = char(raw');
    
    % Close the file
    fclose(fid);

    % Parse JSON string
    data = jsondecode(str);
end