
# Generated: 2025-08-27T14:35:28.437271
# Source Brief: brief_00730.md
# Brief Index: 730

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the miner. Push crystals into the target locations."
    )

    game_description = (
        "An isometric puzzle game. Navigate a miner through a crystal cavern, "
        "shifting crystals to match target patterns within a limited number of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_MOVES_PER_LEVEL = 30
        self.MAX_EPISODE_STEPS = 5 * self.MAX_MOVES_PER_LEVEL + 5 # 5 levels, plus level transitions

        # --- Colors ---
        self.COLOR_BG = (44, 62, 80)
        self.COLOR_GRID_LINE = (52, 73, 94)
        self.COLOR_WALL_SIDE = (127, 140, 141)
        self.COLOR_WALL_TOP = (149, 165, 166)
        self.COLOR_PLAYER = (241, 196, 15)
        self.COLOR_TARGET = (231, 76, 60)
        self.CRYSTAL_COLORS = [
            ((26, 188, 156), (22, 160, 133)), # Cyan (top, side)
            ((155, 89, 182), (142, 68, 173)), # Magenta
            ((230, 126, 34), (211, 84, 0)),   # Orange
            ((52, 152, 219), (41, 128, 185)),  # Blue
        ]
        self.COLOR_UI_TEXT = (236, 240, 241)

        # --- Isometric Grid ---
        self.TILE_WIDTH = 48
        self.TILE_HEIGHT = self.TILE_WIDTH // 2
        self.TILE_DEPTH = 32

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # --- Level Data ---
        self.levels = self._define_levels()
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level_index = 0
        self.moves_left = 0
        self.grid = []
        self.grid_w, self.grid_h = 0, 0
        self.player_pos = (0, 0)
        self.crystals = []
        self.targets = []
        self.grid_origin = (0, 0)
        self.last_action_reward = 0

        self.reset()
        self.validate_implementation()

    def _define_levels(self):
        return [
            { # Level 1: Simple push
                "layout": [
                    "##########",
                    "#        #",
                    "#   P    #",
                    "#   C    #",
                    "#   T    #",
                    "#        #",
                    "##########",
                ]
            },
            { # Level 2: Move one to get to other
                "layout": [
                    "##########",
                    "# P C T  #",
                    "##########",
                ]
            },
            { # Level 3: "S" bend
                "layout": [
                    "###########",
                    "#P  #     #",
                    "# C # T   #",
                    "#   ##### #",
                    "#         #",
                    "###########",
                ]
            },
            { # Level 4: Two crystals, requires ordering
                "layout": [
                    "###########",
                    "#P C1     #",
                    "#  C2 T1  #",
                    "#     T2  #",
                    "###########",
                ]
            },
            { # Level 5: Puzzle box
                "layout": [
                    "############",
                    "#P C1C2    #",
                    "#  #  # T1 #",
                    "# C3# # T2 #",
                    "#   # # T3 #",
                    "#   #      #",
                    "############",
                ]
            },
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level_index = 0
        self._load_level(self.current_level_index)
        return self._get_observation(), self._get_info()

    def _load_level(self, level_index):
        if level_index >= len(self.levels):
            return # Should not happen if game ends properly
        
        level_data = self.levels[level_index]
        layout = level_data["layout"]
        
        self.moves_left = self.MAX_MOVES_PER_LEVEL
        self.grid_h = len(layout)
        self.grid_w = len(layout[0])
        self.grid = [[c for c in row] for row in layout]
        
        self.crystals = []
        self.targets = []
        
        crystal_map = {}
        target_map = {}

        for y, row in enumerate(layout):
            for x, char in enumerate(row):
                if char == 'P':
                    self.player_pos = (x, y)
                    self.grid[y][x] = ' '
                elif char.isdigit():
                    if self.grid[y][x-1] == 'C':
                        crystal_id = int(char)
                        crystal_map[crystal_id] = {'pos': (x, y), 'id': crystal_id}
                        self.grid[y][x] = ' '
                        self.grid[y][x-1] = ' '
                    elif self.grid[y][x-1] == 'T':
                        target_id = int(char)
                        target_map[target_id] = (x,y)
                        self.grid[y][x] = ' '
                        self.grid[y][x-1] = ' '
                elif char == 'C':
                    crystal_map[0] = {'pos': (x,y), 'id': 0}
                    self.grid[y][x] = ' '
                elif char == 'T':
                    target_map[0] = (x,y)
                    self.grid[y][x] = ' '

        # Ensure consistent ordering
        for i in sorted(crystal_map.keys()):
            self.crystals.append(crystal_map[i])
        for i in sorted(target_map.keys()):
            self.targets.append(target_map[i])

        # Center the grid on screen
        grid_pixel_width = (self.grid_w + self.grid_h) * self.TILE_WIDTH / 2
        self.grid_origin = ((self.WIDTH - grid_pixel_width) / 2 + self.grid_h * self.TILE_WIDTH / 2, 
                            self.HEIGHT * 0.2)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = 0
        move_made = False

        if movement != 0: # 0 is no-op
            initial_distances = self._calculate_crystal_distances()
            move_made = self._handle_movement(movement)
            
            if move_made:
                final_distances = self._calculate_crystal_distances()
                distance_change = sum(initial_distances) - sum(final_distances)
                reward += distance_change * 0.1 # Distance-based reward
                self.last_action_reward = distance_change * 0.1
            else:
                self.last_action_reward = 0

        # Check for level completion
        if move_made and self._check_level_complete():
            # SFX: LEVEL_COMPLETE
            reward += 5
            self.score += self.moves_left # Bonus for efficiency
            self.current_level_index += 1
            
            if self.current_level_index >= len(self.levels):
                # Game Won
                self.game_over = True
                reward += 50
            else:
                # Load next level
                self._load_level(self.current_level_index)

        self.steps += 1
        terminated = self._check_termination()
        
        # Apply penalty for losing
        if terminated and not self.game_over: # Lost by running out of moves
            reward -= 10
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement_action):
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement_action]
        px, py = self.player_pos
        nx, ny = px + dx, py + dy

        if not (0 <= nx < self.grid_w and 0 <= ny < self.grid_h) or self.grid[ny][nx] == '#':
            return False # Moved into a wall

        crystal_to_push_idx = self._get_crystal_at((nx, ny))
        if crystal_to_push_idx is None:
            # Simple move
            self.player_pos = (nx, ny)
            self.moves_left -= 1
            # SFX: FOOTSTEP
            return True
        else:
            # Push crystal
            cx, cy = self.crystals[crystal_to_push_idx]['pos']
            slide_x, slide_y = cx, cy
            
            while True:
                next_x, next_y = slide_x + dx, slide_y + dy
                if not (0 <= next_x < self.grid_w and 0 <= next_y < self.grid_h) or \
                   self.grid[next_y][next_x] == '#' or \
                   self._get_crystal_at((next_x, next_y)) is not None:
                    break # Obstacle found
                slide_x, slide_y = next_x, next_y
            
            if (slide_x, slide_y) == (cx, cy):
                return False # Crystal is blocked

            # Execute push
            self.crystals[crystal_to_push_idx]['pos'] = (slide_x, slide_y)
            self.player_pos = (nx, ny)
            self.moves_left -= 1
            # SFX: CRYSTAL_SLIDE
            return True

    def _calculate_crystal_distances(self):
        distances = []
        for i, crystal in enumerate(self.crystals):
            c_pos = crystal['pos']
            t_pos = self.targets[i]
            distances.append(abs(c_pos[0] - t_pos[0]) + abs(c_pos[1] - t_pos[1]))
        return distances

    def _check_level_complete(self):
        return all(c['pos'] == t for c, t in zip(self.crystals, self.targets))

    def _check_termination(self):
        if self.game_over:
            return True
        if self.moves_left <= 0 and not self._check_level_complete():
            return True
        if self.steps >= self.MAX_EPISODE_STEPS:
            return True
        return False

    def _get_crystal_at(self, pos):
        for i, crystal in enumerate(self.crystals):
            if crystal['pos'] == pos:
                return i
        return None

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.current_level_index + 1}

    def _iso_to_screen(self, x, y):
        screen_x = self.grid_origin[0] + (x - y) * self.TILE_WIDTH / 2
        screen_y = self.grid_origin[1] + (x + y) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, grid_pos, top_color, side_color, height):
        x, y = grid_pos
        sx, sy = self._iso_to_screen(x, y)

        # Vertices of the top face
        p_top = (sx, sy - height)
        p_left = (sx - self.TILE_WIDTH // 2, sy - height + self.TILE_HEIGHT // 2)
        p_right = (sx + self.TILE_WIDTH // 2, sy - height + self.TILE_HEIGHT // 2)
        p_bottom = (sx, sy - height + self.TILE_HEIGHT)
        
        top_face = [p_left, p_top, p_right, p_bottom]
        
        # Draw side faces
        pygame.gfxdraw.filled_polygon(surface, [p_left, p_bottom, (p_bottom[0], p_bottom[1] + height), (p_left[0], p_left[1] + height)], side_color)
        pygame.gfxdraw.filled_polygon(surface, [p_right, p_bottom, (p_bottom[0], p_bottom[1] + height), (p_right[0], p_right[1] + height)], side_color)
        
        # Draw top face
        pygame.gfxdraw.filled_polygon(surface, top_face, top_color)
        pygame.gfxdraw.aapolygon(surface, top_face, (0,0,0,100))

    def _render_game(self):
        # Create a sorted list of all objects to draw for correct Z-ordering
        draw_queue = []
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                # Floor and target tiles
                draw_queue.append({'type': 'floor', 'pos': (x, y)})
                if (x, y) in self.targets:
                    draw_queue.append({'type': 'target', 'pos': (x, y)})
                # Walls
                if self.grid[y][x] == '#':
                    draw_queue.append({'type': 'wall', 'pos': (x, y)})
        
        # Add dynamic objects
        for i, crystal in enumerate(self.crystals):
            draw_queue.append({'type': 'crystal', 'pos': crystal['pos'], 'id': crystal['id']})
        
        # Add player with a bobbing animation
        player_bob = math.sin(pygame.time.get_ticks() * 0.005) * 3
        draw_queue.append({'type': 'player', 'pos': self.player_pos, 'bob': player_bob})

        # Sort by Z-index (y+x) and then by object type to ensure player/crystals are on top of floors
        type_order = {'floor': 0, 'target': 1, 'wall': 2, 'crystal': 3, 'player': 4}
        draw_queue.sort(key=lambda item: (item['pos'][0] + item['pos'][1], type_order[item['type']]))

        # --- Drawing ---
        for item in draw_queue:
            pos = item['pos']
            sx, sy = self._iso_to_screen(pos[0], pos[1])
            
            if item['type'] == 'floor':
                points = [
                    (sx, sy), (sx + self.TILE_WIDTH // 2, sy + self.TILE_HEIGHT // 2),
                    (sx, sy + self.TILE_HEIGHT), (sx - self.TILE_WIDTH // 2, sy + self.TILE_HEIGHT // 2)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GRID_LINE)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_BG)

            elif item['type'] == 'target':
                points = [
                    (sx, sy), (sx + self.TILE_WIDTH // 2, sy + self.TILE_HEIGHT // 2),
                    (sx, sy + self.TILE_HEIGHT), (sx - self.TILE_WIDTH // 2, sy + self.TILE_HEIGHT // 2)
                ]
                target_color = (*self.COLOR_TARGET, 150) # Add alpha
                pygame.gfxdraw.filled_polygon(self.screen, points, target_color)

            elif item['type'] == 'wall':
                self._draw_iso_cube(self.screen, pos, self.COLOR_WALL_TOP, self.COLOR_WALL_SIDE, self.TILE_DEPTH)

            elif item['type'] == 'crystal':
                crystal_id = item.get('id', 0)
                colors = self.CRYSTAL_COLORS[crystal_id % len(self.CRYSTAL_COLORS)]
                self._draw_iso_cube(self.screen, pos, colors[0], colors[1], self.TILE_DEPTH)

            elif item['type'] == 'player':
                player_sx, player_sy = self._iso_to_screen(pos[0], pos[1])
                player_sy -= int(item['bob'])
                pygame.draw.circle(self.screen, self.COLOR_PLAYER, (player_sx, player_sy - 10), 10)
                pygame.gfxdraw.aacircle(self.screen, player_sx, player_sy - 10, 10, self.COLOR_PLAYER)
                # Simple shadow
                pygame.gfxdraw.filled_ellipse(self.screen, sx, sy + self.TILE_HEIGHT // 2, 10, 5, (0,0,0,100))


    def _render_ui(self):
        # --- Main UI Panel ---
        level_text = self.font_main.render(f"Level: {self.current_level_index + 1}", True, self.COLOR_UI_TEXT)
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)

        self.screen.blit(level_text, (15, 15))
        self.screen.blit(moves_text, (15, 45))
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 15))

        # --- Target Pattern Preview ---
        preview_w, preview_h = 100, 80
        preview_surf = pygame.Surface((preview_w, preview_h), pygame.SRCALPHA)
        preview_surf.fill((0,0,0,80))
        
        preview_label = self.font_small.render("Target", True, self.COLOR_UI_TEXT)
        preview_surf.blit(preview_label, (preview_w // 2 - preview_label.get_width() // 2, 5))
        
        if self.targets:
            min_tx = min(t[0] for t in self.targets)
            min_ty = min(t[1] for t in self.targets)
            max_tx = max(t[0] for t in self.targets)
            max_ty = max(t[1] for t in self.targets)
            
            t_w = max_tx - min_tx + 1
            t_h = max_ty - min_ty + 1
            
            cell_size = min((preview_w - 20) / t_w, (preview_h - 30) / t_h)
            offset_x = (preview_w - t_w * cell_size) / 2
            offset_y = 25 + (preview_h - 25 - t_h * cell_size) / 2
            
            for i, t_pos in enumerate(self.targets):
                rx, ry = t_pos[0] - min_tx, t_pos[1] - min_ty
                color = self.CRYSTAL_COLORS[i % len(self.CRYSTAL_COLORS)][0]
                rect = pygame.Rect(offset_x + rx * cell_size, offset_y + ry * cell_size, cell_size, cell_size)
                pygame.draw.rect(preview_surf, color, rect)
                pygame.draw.rect(preview_surf, (255,255,255), rect, 1)

        self.screen.blit(preview_surf, (self.WIDTH - preview_w - 10, self.HEIGHT - preview_h - 10))

        # --- Game Over / Win Message ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            if self._check_level_complete():
                msg = "YOU WIN!"
                # SFX: GAME_WIN
            else:
                msg = "GAME OVER"
                # SFX: GAME_LOSE
            
            win_text = pygame.font.Font(None, 72).render(msg, True, self.COLOR_UI_TEXT)
            win_rect = win_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            
            score_final_text = self.font_main.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            score_rect = score_final_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 30))

            overlay.blit(win_text, win_rect)
            overlay.blit(score_final_text, score_rect)
            self.screen.blit(overlay, (0,0))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Crystal Cavern")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop for human play
    running = True
    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    terminated = False
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated and action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
            
        # Get the observation from the environment
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        
        # Draw the frame to the screen
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()