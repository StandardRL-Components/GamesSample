
# Generated: 2025-08-27T16:41:52.418880
# Source Brief: brief_01294.md
# Brief Index: 1294

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Shift to cycle crystal type (R/G/B/Remove). Space to place."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Redirect laser beams using colored crystals to hit the target before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_W, GRID_H = 32, 20
    CELL_W, CELL_H = WIDTH // GRID_W, HEIGHT // GRID_H
    MAX_STEPS = 30 * 60  # 60 seconds at 30fps

    # --- Colors ---
    COLOR_BG = (15, 18, 28)
    COLOR_GRID = (30, 35, 50)
    COLOR_TARGET = (255, 215, 0)
    COLOR_CURSOR = (255, 255, 255)
    
    CRYSTAL_COLORS = {
        1: {'main': (255, 20, 80), 'glow': (255, 50, 120)}, # Red
        2: {'main': (20, 255, 150), 'glow': (50, 255, 180)}, # Green
        3: {'main': (20, 150, 255), 'glow': (50, 180, 255)}, # Blue
    }
    LASER_COLORS = {
        1: {'main': (255, 100, 150), 'glow': (255, 20, 80)}, # Red
        2: {'main': (100, 255, 200), 'glow': (20, 255, 150)}, # Green
        3: {'main': (100, 200, 255), 'glow': (20, 150, 255)}, # Blue
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Etc...        
        
        # Initialize state variables
        self.game_over = False
        self.steps = 0
        self.score = 0
        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        self.cursor_pos = [0, 0]
        self.selected_crystal_type = 1
        self.lasers = []
        self.target_pos = (0, 0)
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.reset()
        # self.validate_implementation() # Uncomment for local testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS
        
        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_crystal_type = 1
        self.particles = []
        
        # Procedural Puzzle Generation
        self.target_pos = (
            self.np_random.integers(self.GRID_W - 5, self.GRID_W - 2),
            self.np_random.integers(5, self.GRID_H - 5)
        )
        
        laser_defs = [
            {'color_id': 1, 'pos': (0, self.np_random.integers(2, self.GRID_H-2)), 'dir': (1, 0)},
            {'color_id': 2, 'pos': (self.np_random.integers(2, self.GRID_W-2), 0), 'dir': (0, 1)},
            {'color_id': 3, 'pos': (self.GRID_W-1, self.np_random.integers(2, self.GRID_H-2)), 'dir': (-1, 0)},
        ]
        self.np_random.shuffle(laser_defs)

        self.lasers = []
        for i in range(self.np_random.integers(2, 4)): # 2 or 3 lasers
            laser_def = laser_defs[i]
            self.lasers.append({
                'color_id': laser_def['color_id'],
                'source_pos': laser_def['pos'],
                'source_dir': laser_def['dir'],
                'path_nodes': [],
                'on_target': False,
            })
            
        self._update_laser_paths()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # --- Handle Input ---
        action_taken = self._handle_input(movement, space_held, shift_held)

        # --- Update Game Logic ---
        self.timer -= 1
        self.steps += 1
        self._update_particles()
        
        if action_taken:
            self._update_laser_paths()
            # Sound effect placeholder
            # play_sound('crystal_place')

        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        action_taken = False
        
        # Cursor movement
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # Cycle crystal type (on press)
        if shift_held and not self.prev_shift_held:
            self.selected_crystal_type = (self.selected_crystal_type % 4) + 1

        # Place/Remove crystal (on press)
        if space_held and not self.prev_space_held:
            cx, cy = self.cursor_pos
            if (cx, cy) != self.target_pos and not any(l['source_pos'] == (cx, cy) for l in self.lasers):
                if self.selected_crystal_type == 4: # Remove
                    self.grid[cx, cy] = 0
                else:
                    self.grid[cx, cy] = self.selected_crystal_type
                self._spawn_placement_effect(cx, cy)
                action_taken = True
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        return action_taken

    def _update_laser_paths(self):
        for laser in self.lasers:
            laser['path_nodes'] = [laser['source_pos']]
            laser['on_target'] = False
            
            pos = laser['source_pos']
            direction = laser['source_dir']
            
            for _ in range(self.GRID_W * self.GRID_H): # Max path length
                next_pos = (pos[0] + direction[0], pos[1] + direction[1])
                
                if next_pos == self.target_pos:
                    laser['path_nodes'].append(next_pos)
                    laser['on_target'] = True
                    # Sound effect placeholder
                    # play_sound('laser_hit_target')
                    break
                    
                if not (0 <= next_pos[0] < self.GRID_W and 0 <= next_pos[1] < self.GRID_H):
                    laser['path_nodes'].append(next_pos) # For drawing to edge
                    break
                
                crystal_type = self.grid[next_pos[0], next_pos[1]]
                if crystal_type == laser['color_id']:
                    pos = next_pos
                    laser['path_nodes'].append(pos)
                    # 90 degree clockwise reflection
                    direction = (-direction[1], direction[0])
                    # Sound effect placeholder
                    # play_sound('laser_reflect')
                else: # Empty or wrong color
                    pos = next_pos
            
            # If the path is just a straight line, add the final point
            if len(laser['path_nodes']) == 1:
                laser['path_nodes'].append(pos)

    def _calculate_reward(self):
        reward = 0
        
        for laser in self.lasers:
            if laser['on_target']:
                reward += 10
            else:
                # Proximity/Angle reward
                if len(laser['path_nodes']) > 1:
                    # Use last two nodes to determine final direction
                    p_last = self._grid_to_screen(laser['path_nodes'][-1])
                    p_prev = self._grid_to_screen(laser['path_nodes'][-2])
                    
                    vec_laser = np.array([p_last[0] - p_prev[0], p_last[1] - p_prev[1]])
                    vec_target = np.array([self._grid_to_screen(self.target_pos)[0] - p_last[0],
                                           self._grid_to_screen(self.target_pos)[1] - p_last[1]])
                    
                    norm_laser = np.linalg.norm(vec_laser)
                    norm_target = np.linalg.norm(vec_target)
                    
                    if norm_laser > 0 and norm_target > 0:
                        cos_sim = np.dot(vec_laser, vec_target) / (norm_laser * norm_target)
                        reward += max(0, cos_sim) * 0.01

        return reward

    def _check_termination(self):
        win_condition = all(l['on_target'] for l in self.lasers)
        if win_condition:
            self.score += 100
            self.game_over = True
            return True
        
        if self.timer <= 0:
            self.score -= 100
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
        }

    def _grid_to_screen(self, pos):
        x, y = pos
        return int(x * self.CELL_W + self.CELL_W / 2), int(y * self.CELL_H + self.CELL_H / 2)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_W + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.CELL_W, 0), (x * self.CELL_W, self.HEIGHT))
        for y in range(self.GRID_H + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.CELL_H), (self.WIDTH, y * self.CELL_H))

        # Draw target
        tx, ty = self._grid_to_screen(self.target_pos)
        self._draw_glow_circle(tx, ty, int(self.CELL_W * 0.4), self.COLOR_TARGET, (255, 255, 100))

        # Draw crystals
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                if self.grid[x, y] != 0:
                    self._draw_crystal((x, y), self.grid[x, y])
        
        # Draw lasers
        for laser in self.lasers:
            colors = self.LASER_COLORS[laser['color_id']]
            path_pixels = [self._grid_to_screen(p) for p in laser['path_nodes']]
            if len(path_pixels) > 1:
                self._draw_glow_path(path_pixels, colors['glow'], colors['main'])

        # Draw laser sources
        for laser in self.lasers:
            sx, sy = self._grid_to_screen(laser['source_pos'])
            color = self.LASER_COLORS[laser['color_id']]['main']
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, 5, color)
            pygame.gfxdraw.aacircle(self.screen, sx, sy, 5, color)

    def _render_ui(self):
        # Timer
        timer_text = f"TIME: {self.timer / 30:.1f}"
        text_surf = self.font_small.render(timer_text, True, (200, 200, 220))
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))

        # Selected Crystal UI
        ui_x, ui_y = 15, self.HEIGHT - 40
        for i in range(1, 5):
            is_selected = self.selected_crystal_type == i
            rect = pygame.Rect(ui_x, ui_y, 30, 30)
            
            if is_selected:
                pygame.draw.rect(self.screen, (255,255,255), rect, 2, 3)

            if i <= 3: # Crystal
                colors = self.CRYSTAL_COLORS[i]
                points = [
                    (rect.centerx, rect.top + 3),
                    (rect.right - 3, rect.centery),
                    (rect.centerx, rect.bottom - 3),
                    (rect.left + 3, rect.centery),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, colors['main'])
                pygame.gfxdraw.filled_polygon(self.screen, points, colors['glow'] + (100,))
            else: # Remove icon
                pygame.draw.line(self.screen, (255, 80, 80), rect.topleft, rect.bottomright, 2)
                pygame.draw.line(self.screen, (255, 80, 80), rect.topright, rect.bottomleft, 2)
            
            ui_x += 40

        # Cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(cx * self.CELL_W, cy * self.CELL_H, self.CELL_W, self.CELL_H)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 1)

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = all(l['on_target'] for l in self.lasers)
            msg = "LEVEL CLEAR" if win_condition else "TIME UP"
            color = (150, 255, 150) if win_condition else (255, 150, 150)
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _draw_crystal(self, pos, type_id):
        px, py = self._grid_to_screen(pos)
        colors = self.CRYSTAL_COLORS[type_id]
        
        points = [
            (px, py - self.CELL_H // 3),
            (px + self.CELL_W // 3, py),
            (px, py + self.CELL_H // 3),
            (px - self.CELL_W // 3, py),
        ]
        
        pygame.gfxdraw.filled_polygon(self.screen, points, colors['glow'])
        pygame.gfxdraw.aapolygon(self.screen, points, colors['main'])
        pygame.gfxdraw.filled_polygon(self.screen, points, colors['main'])

    def _draw_glow_circle(self, x, y, radius, color_glow, color_main):
        for i in range(5, 0, -1):
            alpha = 40 - i * 5
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius + i, color_glow + (alpha,))
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color_main)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color_main)

    def _draw_glow_path(self, points, color_glow, color_main):
        try:
            pygame.draw.aalines(self.screen, color_glow, False, points, 3)
            pygame.draw.aalines(self.screen, color_main, False, points, 1)
        except:
             # Pygame can fail on zero-length lines
             pass

    def _spawn_placement_effect(self, gx, gy):
        px, py = self._grid_to_screen((gx, gy))
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'x': px, 'y': py,
                'dx': math.cos(angle) * speed, 'dy': math.sin(angle) * speed,
                'life': self.np_random.integers(15, 25),
                'color': (200, 200, 255)
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['dx']
            p['y'] += p['dy']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            size = max(0, p['life'] // 5)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (p['x'], p['y']), size)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("✓ Running implementation validation...")
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a display for human playing
    pygame.display.set_caption("Crystal Cavern")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Space
        if keys[pygame.K_SPACE]: action[1] = 1
        
        # Shift
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation from the environment to the display screen
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            env.reset()

    env.close()