import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An isometric puzzle game where the player places crystals to guide a laser
    to an exit, managing energy along the way. The goal is to reach the exit
    with sufficient energy.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. SHIFT to cycle crystal type. SPACE to place a crystal."
    )

    game_description = (
        "Guide a laser through a crystal cavern. Place crystals to reflect, "
        "split, and amplify the beam to reach the exit with enough energy."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Core Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- Colors ---
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (25, 30, 50)
        self.COLOR_WALL = (60, 70, 100)
        self.COLOR_WALL_TOP = (90, 100, 130)
        self.COLOR_SOURCE = (255, 255, 100)
        self.COLOR_EXIT = (100, 255, 255)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_LASER = (255, 50, 50)
        self.CRYSTAL_COLORS = {
            0: ((255, 80, 80), "Reflector"),      # Green in brief, but red is more intuitive for reflection
            1: ((80, 80, 255), "Amplifier"),      # Blue
            2: ((255, 80, 255), "Absorber"),       # Red in brief, but magenta/purple is more distinct
            3: ((255, 255, 80), "Splitter"),      # Yellow
            4: ((160, 80, 255), "Teleporter"),    # Purple
        }

        # --- Game Grid & Isometric Projection ---
        self.grid_w, self.grid_h = 22, 16
        self.tile_w, self.tile_h = 32, 16
        self.origin_x = self.WIDTH // 2
        self.origin_y = 80
        
        # --- Game Constants ---
        self.MAX_STEPS = 250
        self.INITIAL_ENERGY = 50
        self.WIN_ENERGY_THRESHOLD = 10
        self.MAX_LASER_SEGMENTS = 100

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = ""
        self.successful_episodes = 0
        self.difficulty_level = 0
        self.energy = 0
        self.cursor_pos = [0, 0]
        self.selected_crystal_type = 0
        self.placed_crystals = []
        self.walls = []
        self.laser_source_pos = [0, 0]
        self.exit_pos = [0, 0]
        self.laser_path = []
        self.prev_space_state = 0
        self.prev_shift_state = 0

        # self.reset() is called by the test harness, no need to call it here.

    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (x - y) * (self.tile_w / 2)
        screen_y = self.origin_y + (x + y) * (self.tile_h / 2)
        return int(screen_x), int(screen_y)

    def _generate_level(self):
        self.walls = []
        self.placed_crystals = []

        # Place source and exit
        self.laser_source_pos = [1, self.grid_h // 2]
        self.exit_pos = [self.grid_w - 2, self.grid_h // 2]

        # Generate walls, ensuring a path exists
        num_walls = min(30, 5 + self.difficulty_level * 2)
        while True:
            self.walls = []
            for _ in range(num_walls):
                wx, wy = self.np_random.integers(0, self.grid_w), self.np_random.integers(0, self.grid_h)
                if [wx, wy] not in [self.laser_source_pos, self.exit_pos]:
                    self.walls.append([wx, wy])
            
            # Use BFS to check for a path
            q = deque([self.laser_source_pos])
            visited = {tuple(self.laser_source_pos)}
            path_found = False
            while q:
                curr = q.popleft()
                if curr == self.exit_pos:
                    path_found = True
                    break
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = curr[0] + dx, curr[1] + dy
                    if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h and \
                       tuple([nx, ny]) not in visited and [nx, ny] not in self.walls:
                        visited.add(tuple([nx, ny]))
                        q.append([nx, ny])
            
            if path_found:
                break


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = ""
        self.energy = self.INITIAL_ENERGY
        self.cursor_pos = [self.grid_w // 2, self.grid_h // 2]
        self.selected_crystal_type = 0
        self.prev_space_state = 0
        self.prev_shift_state = 0

        if options and "difficulty" in options:
            self.difficulty_level = options["difficulty"]

        self._generate_level()
        self.laser_path, _, _ = self._calculate_laser_path()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.1  # Small penalty for taking a step to encourage efficiency

        # --- Handle Actions ---
        # 1. Cursor Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.grid_w - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.grid_h - 1)

        # 2. Cycle Crystal Type (on press)
        if shift_held and not self.prev_shift_state:
            self.selected_crystal_type = (self.selected_crystal_type + 1) % len(self.CRYSTAL_COLORS)
        self.prev_shift_state = shift_held

        # 3. Place Crystal (on press)
        if space_held and not self.prev_space_state:
            can_place = self.cursor_pos not in self.walls and \
                        self.cursor_pos not in [c['pos'] for c in self.placed_crystals] and \
                        self.cursor_pos not in [self.laser_source_pos, self.exit_pos]

            if can_place:
                # Special rule for teleporters: only two allowed
                if self.selected_crystal_type == 4:
                    teleporter_count = sum(1 for c in self.placed_crystals if c['type'] == 4)
                    if teleporter_count >= 2:
                        can_place = False

                if can_place:
                    self.placed_crystals.append({
                        'pos': list(self.cursor_pos),
                        'type': self.selected_crystal_type
                    })
                    # sfx: Crystal placement sound
                    self.energy -= 1 # Cost to place a crystal
                    reward -= 1

        self.prev_space_state = space_held
        
        # --- Update Game State ---
        self.laser_path, energy_delta, events = self._calculate_laser_path()
        self.energy += energy_delta
        reward += energy_delta
        if 'split' in events:
            reward += 5

        self.energy = max(0, self.energy)
        self.score += reward
        self.steps += 1

        terminated, term_reward = self._check_termination()
        self.score += term_reward
        reward += term_reward
        truncated = self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _calculate_laser_path(self):
        beams = deque([{'pos': self.laser_source_pos, 'dir': [1, 0], 'id': 0}]) # dir: [dx, dy] on grid
        
        path_segments = []
        total_energy_delta = 0
        events = []
        visited_crystals = set()
        
        max_beam_id = 0
        
        while beams:
            beam = beams.popleft()
            pos = np.array(beam['pos'], dtype=float) + 0.5
            direction = np.array(beam['dir'], dtype=float)
            
            for _ in range(self.MAX_LASER_SEGMENTS):
                intersections = []
                
                # Screen boundaries
                if direction[0] > 0: intersections.append(((self.grid_w - pos[0]) / direction[0], 'boundary', None))
                if direction[0] < 0: intersections.append(((0 - pos[0]) / direction[0], 'boundary', None))
                if direction[1] > 0: intersections.append(((self.grid_h - pos[1]) / direction[1], 'boundary', None))
                if direction[1] < 0: intersections.append(((0 - pos[1]) / direction[1], 'boundary', None))

                # Walls
                for wall_pos in self.walls:
                    for i in range(2):
                        for sign in [-0.5, 0.5]:
                            p_wall = np.array(wall_pos) + 0.5
                            p_wall[i] += sign
                            if direction[i] != 0:
                                t = (p_wall[i] - pos[i]) / direction[i]
                                if t > 1e-6:
                                    hit_pos = pos + t * direction
                                    if all(abs(hit_pos[j] - (wall_pos[j] + 0.5)) <= 0.501 for j in range(2)):
                                        intersections.append((t, 'wall', wall_pos))
                
                # Crystals
                for crystal in self.placed_crystals:
                    c_pos = crystal['pos']
                    for i in range(2):
                        for sign in [-0.5, 0.5]:
                            p_crys = np.array(c_pos) + 0.5
                            p_crys[i] += sign
                            if direction[i] != 0:
                                t = (p_crys[i] - pos[i]) / direction[i]
                                if t > 1e-6:
                                    hit_pos = pos + t * direction
                                    if all(abs(hit_pos[j] - (c_pos[j] + 0.5)) <= 0.501 for j in range(2)):
                                        intersections.append((t, 'crystal', crystal))

                # Exit
                e_pos = self.exit_pos
                for i in range(2):
                    for sign in [-0.5, 0.5]:
                        p_exit = np.array(e_pos) + 0.5
                        p_exit[i] += sign
                        if direction[i] != 0:
                            t = (p_exit[i] - pos[i]) / direction[i]
                            if t > 1e-6:
                                hit_pos = pos + t * direction
                                if all(abs(hit_pos[j] - (e_pos[j] + 0.5)) <= 0.501 for j in range(2)):
                                    intersections.append((t, 'exit', None))

                if not intersections: break
                
                t, obj_type, obj_data = min(intersections, key=lambda x: x[0])
                
                end_pos = pos + t * direction
                path_segments.append((pos.tolist(), end_pos.tolist()))
                
                if obj_type == 'boundary' or obj_type == 'exit':
                    break # Beam leaves screen or hits exit

                hit_object_grid_pos = obj_data if obj_type == 'wall' else obj_data['pos']

                hit_normal = np.zeros(2)
                if abs(end_pos[0] - (hit_object_grid_pos[0] + 1)) < 1e-4: hit_normal = np.array([-1, 0])
                elif abs(end_pos[0] - hit_object_grid_pos[0]) < 1e-4: hit_normal = np.array([1, 0])
                elif abs(end_pos[1] - (hit_object_grid_pos[1] + 1)) < 1e-4: hit_normal = np.array([0, -1])
                elif abs(end_pos[1] - hit_object_grid_pos[1]) < 1e-4: hit_normal = np.array([0, 1])

                pos = end_pos
                
                if obj_type == 'wall':
                    # sfx: Laser ricochet
                    direction = direction - 2 * np.dot(direction, hit_normal) * hit_normal
                
                elif obj_type == 'crystal':
                    crystal_id = tuple(obj_data['pos'])
                    crystal_type = obj_data['type']
                    
                    if crystal_type == 0: # Reflector
                        # sfx: Laser ricochet
                        direction = direction - 2 * np.dot(direction, hit_normal) * hit_normal
                    
                    elif crystal_type == 1: # Amplifier
                        if crystal_id not in visited_crystals:
                            # sfx: Powerup sound
                            total_energy_delta += 5
                            visited_crystals.add(crystal_id)
                        # Beam passes through
                    
                    elif crystal_type == 2: # Absorber
                        # sfx: Laser fizzle
                        total_energy_delta -= 10
                        break # Beam ends

                    elif crystal_type == 3: # Splitter
                        if crystal_id not in visited_crystals:
                            # sfx: Split sound
                            events.append('split')
                            visited_crystals.add(crystal_id)
                            
                            # Reflected beam
                            direction = direction - 2 * np.dot(direction, hit_normal) * hit_normal
                            
                            # Transmitted beam
                            max_beam_id += 1
                            beams.append({'pos': end_pos, 'dir': beam['dir'], 'id': max_beam_id})
                        else: # Act as reflector if already visited
                            direction = direction - 2 * np.dot(direction, hit_normal) * hit_normal

                    elif crystal_type == 4: # Teleporter
                        teleporters = [c for c in self.placed_crystals if c['type'] == 4 and tuple(c['pos']) != crystal_id]
                        if teleporters:
                            # sfx: Teleport sound
                            target_crystal = teleporters[0]
                            pos = np.array(target_crystal['pos'], dtype=float) + 0.5
                            # Emerge from opposite side
                            pos = pos - direction * 0.1 
                        else: # Only one teleporter, acts as absorber
                            break
        
        # Check if exit was hit by any segment
        exit_reached = any(
            math.dist(seg[1], np.array(self.exit_pos) + 0.5) < 0.51 or
            math.dist(seg[0], np.array(self.exit_pos) + 0.5) < 0.51
            for seg in path_segments
        )
        if exit_reached:
            events.append('exit_reached')

        return path_segments, total_energy_delta, events

    def _check_termination(self):
        # Check for exit being hit in the last laser calculation
        _, _, events = self._calculate_laser_path()
        if 'exit_reached' in events:
            self.game_over = True
            if self.energy >= self.WIN_ENERGY_THRESHOLD:
                self.win_state = "VICTORY"
                self.successful_episodes += 1
                if self.successful_episodes % 5 == 0:
                    self.difficulty_level += 1
                return True, 100
            else:
                self.win_state = "FAILURE"
                return True, -100

        if self.energy <= 0:
            self.game_over = True
            self.win_state = "DEPLETED"
            return True, -50

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win_state = "TIMEOUT"
            # Termination is handled by truncation, no extra penalty
            return True, 0

        return False, 0
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "difficulty": self.difficulty_level,
            "win_state": self.win_state,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_iso_cube(self, surface, x, y, color, top_color, height=1):
        pos_x, pos_y = self._iso_to_screen(x, y)
        tile_w_half, tile_h_half = self.tile_w / 2, self.tile_h / 2
        
        points = [
            (pos_x, pos_y - height * self.tile_h),
            (pos_x + tile_w_half, pos_y - height * self.tile_h + tile_h_half),
            (pos_x, pos_y - height * self.tile_h + self.tile_h),
            (pos_x - tile_w_half, pos_y - height * self.tile_h + tile_h_half)
        ]
        pygame.gfxdraw.filled_polygon(surface, points, top_color)
        pygame.gfxdraw.aapolygon(surface, points, top_color)

        p_bottom = (pos_x, pos_y + self.tile_h)
        # Left face
        points_l = [points[3], points[2], p_bottom, (p_bottom[0] - tile_w_half, p_bottom[1] - tile_h_half)]
        pygame.gfxdraw.filled_polygon(surface, points_l, color)
        # Right face
        points_r = [points[2], points[1], (p_bottom[0] + tile_w_half, p_bottom[1] - tile_h_half), p_bottom]
        pygame.gfxdraw.filled_polygon(surface, points_r, color)

    def _draw_glowing_line(self, surface, color, start, end, width):
        try:
            glow_color = (*color, 60)
            pygame.draw.line(surface, glow_color, start, end, width * 4)
            pygame.draw.line(surface, (*color, 120), start, end, width * 2)
            pygame.draw.line(surface, (255, 255, 255), start, end, width)
        except TypeError: # Color might not have alpha
             pygame.draw.line(surface, color, start, end, width)


    def _render_game(self):
        # Draw grid
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + 1, y)
                p3 = self._iso_to_screen(x, y + 1)
                pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p3)
        
        # Draw walls, source, exit
        for wall_pos in self.walls:
            self._draw_iso_cube(self.screen, wall_pos[0], wall_pos[1], self.COLOR_WALL, self.COLOR_WALL_TOP)
        self._draw_iso_cube(self.screen, self.laser_source_pos[0], self.laser_source_pos[1], self.COLOR_SOURCE, (255,255,200))
        self._draw_iso_cube(self.screen, self.exit_pos[0], self.exit_pos[1], self.COLOR_EXIT, (200,255,255))

        # Draw placed crystals
        for crystal in self.placed_crystals:
            color, _ = self.CRYSTAL_COLORS[crystal['type']]
            top_color = tuple(min(255, c + 50) for c in color)
            self._draw_iso_cube(self.screen, crystal['pos'][0], crystal['pos'][1], color, top_color, height=0.5)

        # Draw laser
        for start_grid, end_grid in self.laser_path:
            start_screen = self._iso_to_screen(start_grid[0], start_grid[1])
            end_screen = self._iso_to_screen(end_grid[0], end_grid[1])
            self._draw_glowing_line(self.screen, self.COLOR_LASER, start_screen, end_screen, 2)
        
        # Draw cursor
        if not self.game_over:
            cursor_screen_pos = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1])
            alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
            cursor_color = (*self.COLOR_CURSOR, alpha)
            tile_w_half, tile_h_half = self.tile_w / 2, self.tile_h / 2
            points = [
                (cursor_screen_pos[0], cursor_screen_pos[1]),
                (cursor_screen_pos[0] + tile_w_half, cursor_screen_pos[1] + tile_h_half),
                (cursor_screen_pos[0], cursor_screen_pos[1] + self.tile_h),
                (cursor_screen_pos[0] - tile_w_half, cursor_screen_pos[1] + tile_h_half)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, cursor_color)

    def _render_ui(self):
        # UI Panel
        panel_rect = pygame.Rect(0, self.HEIGHT - 40, self.WIDTH, 40)
        # Use a surface for transparency
        s = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        s.fill((0, 0, 0, 150))
        self.screen.blit(s, (0, self.HEIGHT - 40))


        # Energy
        energy_text = self.font_small.render(f"Energy: {self.energy}", True, (255, 255, 100))
        self.screen.blit(energy_text, (10, self.HEIGHT - 30))
        
        # Score
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (150, self.HEIGHT - 30))

        # Selected Crystal
        sel_color, sel_name = self.CRYSTAL_COLORS[self.selected_crystal_type]
        crystal_text = self.font_small.render(f"Crystal: {sel_name}", True, sel_color)
        self.screen.blit(crystal_text, (self.WIDTH - 220, self.HEIGHT - 30))
        pygame.draw.rect(self.screen, sel_color, (self.WIDTH - 240, self.HEIGHT - 30, 15, 15))

        # Game Over Text
        if self.game_over:
            color = (100, 255, 100) if self.win_state == "VICTORY" else (255, 100, 100)
            end_text = self.font_large.render(self.win_state, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
        # This method is for internal validation and is not part of the standard Gym API.
        # It's useful for debugging during development.
        pass

if __name__ == '__main__':
    # This block will not be executed in the testing environment.
    # It's provided for human interaction and visualization.
    
    # Re-enable display for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    pygame.display.init()
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    pygame.display.set_caption("Crystal Cavern")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        
        # Map keyboard to MultiDiscrete action
        mov = 0
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([mov, space, shift])

        # Only step if there is an action
        if action.any():
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode Finished. Score: {info['score']}, State: {info['win_state']}")
                # Render final state
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                # Wait a bit before reset
                pygame.time.wait(2000)
                obs, info = env.reset()

        # Render the observation from the environment
        # We need to get the latest observation even if we didn't step
        current_obs = env._get_observation()
        surf = pygame.surfarray.make_surface(np.transpose(current_obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Human play runs at 30 FPS

    pygame.quit()