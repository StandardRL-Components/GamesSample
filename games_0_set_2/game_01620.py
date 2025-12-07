
# Generated: 2025-08-27T17:43:46.957412
# Source Brief: brief_01620.md
# Brief Index: 1620

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to push the selected crystal. Press Space to cycle which crystal is selected."
    )

    game_description = (
        "Navigate a shimmering crystal maze. Push crystals to activate switches and unlock the exit portal. Solve the puzzle before you run out of moves."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 12, 10
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 28, 14
    CRYSTAL_HEIGHT = 24

    # --- Colors ---
    COLOR_BG = (15, 20, 45)
    COLOR_GRID = (30, 40, 70)
    COLOR_TEXT = (220, 220, 255)

    COLOR_CRYSTAL_RED = (255, 50, 50)
    COLOR_CRYSTAL_BLUE = (50, 150, 255)
    COLOR_CRYSTAL_GREEN = (50, 255, 150)

    COLOR_SWITCH_OFF = (100, 100, 120)
    COLOR_SWITCH_ON = (255, 255, 100)

    COLOR_PORTAL_CLOSED = (150, 50, 255)
    COLOR_PORTAL_OPEN = (255, 255, 255)
    
    COLOR_SELECTOR = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.grid_offset_x = self.SCREEN_WIDTH // 2
        self.grid_offset_y = self.SCREEN_HEIGHT // 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_HALF) // 2 + 30
        
        self.animation_timer = 0
        self.last_space_held = False
        
        self.crystals = []
        self.switches = []
        self.exit_pos = (0, 0)
        self.exit_open = False
        self.movable_crystals = []
        self.selected_crystal_idx = 0
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.animation_timer = 0
        self.last_space_held = False

        self._generate_level()
        
        self.movable_crystals = [
            i for i, c in enumerate(self.crystals) if c['type'] in ['blue', 'green']
        ]
        self.selected_crystal_idx = 0 if self.movable_crystals else -1
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        reward = 0
        
        self.animation_timer += 1

        # --- Action: Select Crystal ---
        space_pressed = space_held and not self.last_space_held
        if space_pressed and self.movable_crystals:
            # sound: select.wav
            self.selected_crystal_idx = (self.selected_crystal_idx + 1) % len(self.movable_crystals)
        self.last_space_held = bool(space_held)

        # --- Action: Move Crystal ---
        if movement > 0 and self.selected_crystal_idx != -1:
            self.steps += 1
            reward -= 0.1
            
            crystal_global_idx = self.movable_crystals[self.selected_crystal_idx]
            crystal = self.crystals[crystal_global_idx]
            
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            
            # Slide logic
            start_pos = crystal['pos']
            current_pos = start_pos
            while True:
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                # Check boundaries
                if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                    break
                
                # Check collision with other crystals
                if any(c['pos'] == next_pos for c in self.crystals):
                    break
                
                current_pos = next_pos
            
            if current_pos != start_pos:
                # sound: crystal_slide.wav
                crystal['pos'] = current_pos

                # --- Post-Move Updates ---
                reward += self._update_switches()
                reward += self._check_exit_condition()

                # Check win condition
                if self.exit_open and crystal['pos'] == self.exit_pos:
                    # sound: win.wav
                    reward += 50
                    self.game_over = True
        
        # --- Termination ---
        if self.steps >= 50: # Turn limit
            if not self.game_over: # Avoid overwriting a win
                # sound: lose.wav
                self.game_over = True

        self.score += reward
        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_level(self):
        self.exit_pos = (self.GRID_WIDTH - 1, self.GRID_HEIGHT // 2)
        self.exit_open = False
        
        self.switches = [
            {'pos': (2, self.GRID_HEIGHT - 2), 'on': False},
            {'pos': (self.GRID_WIDTH - 3, 2), 'on': False},
        ]
        
        self.crystals = [
            # Movable
            {'pos': (3, 4), 'type': 'blue'},
            {'pos': (2, 2), 'type': 'green'},
            {'pos': (self.GRID_WIDTH - 4, 5), 'type': 'green'},
            # Immovable Obstacles
            {'pos': (5, 3), 'type': 'red'}, {'pos': (5, 4), 'type': 'red'}, {'pos': (5, 5), 'type': 'red'},
            {'pos': (0, self.GRID_HEIGHT - 2), 'type': 'red'},
            {'pos': (self.GRID_WIDTH - 3, 0), 'type': 'red'},
        ]

    def _update_switches(self):
        reward = 0
        green_crystals_pos = [c['pos'] for c in self.crystals if c['type'] == 'green']
        
        for switch in self.switches:
            was_on = switch['on']
            is_on = False
            for gc_pos in green_crystals_pos:
                if abs(gc_pos[0] - switch['pos'][0]) + abs(gc_pos[1] - switch['pos'][1]) == 1:
                    is_on = True
                    break
            switch['on'] = is_on
            if is_on and not was_on:
                # sound: switch_on.wav
                reward += 1
        return reward

    def _check_exit_condition(self):
        if not self.exit_open and all(s['on'] for s in self.switches):
            # sound: portal_open.wav
            self.exit_open = True
            return 5
        return 0

    def _cart_to_iso(self, x, y):
        iso_x = self.grid_offset_x + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.grid_offset_y + (x + y) * self.TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, pos, color, height):
        x, y = pos
        center_x, top_y = self._cart_to_iso(x, y)
        
        # Colors for shading
        light_color = tuple(min(255, c + 30) for c in color)
        dark_color = tuple(max(0, c - 30) for c in color)
        
        # Points for polygons
        top_points = [
            (center_x, top_y - self.TILE_HEIGHT_HALF),
            (center_x + self.TILE_WIDTH_HALF, top_y),
            (center_x, top_y + self.TILE_HEIGHT_HALF),
            (center_x - self.TILE_WIDTH_HALF, top_y),
        ]
        
        bottom_y = top_y + height
        
        # Left Side
        pygame.draw.polygon(surface, dark_color, [
            (center_x - self.TILE_WIDTH_HALF, top_y),
            (center_x, top_y + self.TILE_HEIGHT_HALF),
            (center_x, bottom_y + self.TILE_HEIGHT_HALF),
            (center_x - self.TILE_WIDTH_HALF, bottom_y),
        ])
        
        # Right Side
        pygame.draw.polygon(surface, color, [
            (center_x + self.TILE_WIDTH_HALF, top_y),
            (center_x, top_y + self.TILE_HEIGHT_HALF),
            (center_x, bottom_y + self.TILE_HEIGHT_HALF),
            (center_x + self.TILE_WIDTH_HALF, bottom_y),
        ])
        
        # Top Face
        pygame.draw.polygon(surface, light_color, top_points)
        pygame.draw.aalines(surface, (0,0,0,50), True, top_points)

    def _render_game(self):
        # --- Draw Grid ---
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                sx, sy = self._cart_to_iso(x, y)
                points = [
                    (sx, sy - self.TILE_HEIGHT_HALF),
                    (sx + self.TILE_WIDTH_HALF, sy),
                    (sx, sy + self.TILE_HEIGHT_HALF),
                    (sx - self.TILE_WIDTH_HALF, sy),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # --- Collect and Sort All Drawable Entities ---
        renderables = []
        for switch in self.switches:
            renderables.append(('switch', switch))
        for crystal in self.crystals:
            renderables.append(('crystal', crystal))
        renderables.append(('portal', {'pos': self.exit_pos}))
        
        # Sort by depth (y+x) for correct isometric rendering
        renderables.sort(key=lambda item: item[1]['pos'][0] + item[1]['pos'][1])

        # --- Draw Entities ---
        for entity_type, data in renderables:
            pos = data['pos']
            sx, sy = self._cart_to_iso(pos[0], pos[1])
            
            if entity_type == 'switch':
                color = self.COLOR_SWITCH_ON if data['on'] else self.COLOR_SWITCH_OFF
                points = [ (sx, sy - self.TILE_HEIGHT_HALF), (sx + self.TILE_WIDTH_HALF, sy), (sx, sy + self.TILE_HEIGHT_HALF), (sx - self.TILE_WIDTH_HALF, sy) ]
                pygame.draw.polygon(self.screen, color, points)
                
            elif entity_type == 'portal':
                color = self.COLOR_PORTAL_OPEN if self.exit_open else self.COLOR_PORTAL_CLOSED
                glow_intensity = 1 + 0.1 * math.sin(self.animation_timer * 0.1)
                radius = int(self.TILE_WIDTH_HALF * 0.8 * glow_intensity)
                for i in range(radius, 0, -2):
                    alpha = 150 * (1 - i / radius)
                    pygame.gfxdraw.filled_circle(self.screen, sx, sy, i, (*color, int(alpha)))
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, int(radius*0.5), color)

            elif entity_type == 'crystal':
                crystal_color = {
                    'red': self.COLOR_CRYSTAL_RED,
                    'blue': self.COLOR_CRYSTAL_BLUE,
                    'green': self.COLOR_CRYSTAL_GREEN
                }[data['type']]
                self._draw_iso_cube(self.screen, pos, crystal_color, self.CRYSTAL_HEIGHT)

        # --- Draw Selector for the active crystal ---
        if self.selected_crystal_idx != -1:
            crystal_idx = self.movable_crystals[self.selected_crystal_idx]
            selected_pos = self.crystals[crystal_idx]['pos']
            sx, sy = self._cart_to_iso(selected_pos[0], selected_pos[1])
            
            pulse = 1 + 0.15 * math.sin(self.animation_timer * 0.2)
            radius = int(self.TILE_WIDTH_HALF * pulse)
            points = []
            for i in range(3):
                angle = math.radians(120 * i - 90)
                points.append((sx + math.cos(angle) * radius, sy - self.CRYSTAL_HEIGHT - 10 + math.sin(angle) * radius))
            pygame.draw.polygon(self.screen, self.COLOR_SELECTOR, points)


    def _render_ui(self):
        moves_text = self.font_main.render(f"Moves: {max(0, 50 - self.steps)}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))
        
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            is_win = self.exit_open and any(c['pos'] == self.exit_pos for c in self.crystals if c['type'] in ['blue', 'green'])
            
            end_text_str = "PUZZLE SOLVED" if is_win else "OUT OF MOVES"
            end_text = self.font_main.render(end_text_str, True, self.COLOR_PORTAL_OPEN if is_win else self.COLOR_CRYSTAL_RED)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": max(0, 50 - self.steps),
            "exit_open": self.exit_open
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Human Player Controls ---
    # Use arrow keys to move, space to select, q to quit
    
    # To keep track of which keys are held down
    keys_held = {
        pygame.K_UP: False, pygame.K_DOWN: False,
        pygame.K_LEFT: False, pygame.K_RIGHT: False,
        pygame.K_SPACE: False, pygame.K_LSHIFT: False
    }
    
    # Create a window for human play
    pygame.display.set_caption("Crystal Maze")
    window = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        movement_action = 0 # 0=none
        space_action = 0
        
        # Process events once per frame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    terminated = False
                # For turn-based, we only care about the key press event, not holding
                if not terminated:
                    if event.key == pygame.K_UP: movement_action = 1
                    elif event.key == pygame.K_DOWN: movement_action = 2
                    elif event.key == pygame.K_LEFT: movement_action = 3
                    elif event.key == pygame.K_RIGHT: movement_action = 4
                    elif event.key == pygame.K_SPACE: space_action = 1
        
        # Only process one action per frame for turn-based game
        if not terminated and (movement_action > 0 or space_action > 0):
            action = [
                movement_action,
                space_action,
                0, # Shift not used
            ]
            obs, reward, terminated, truncated, info = env.step(action)
        else:
             # If no action, we still need to render the current state
             obs = env._get_observation()

        # Render the observation to the window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        window.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human play

    env.close()