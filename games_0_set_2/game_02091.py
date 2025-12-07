
# Generated: 2025-08-28T03:40:28.822585
# Source Brief: brief_02091.md
# Brief Index: 2091

        
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
        "Controls: Arrow keys to move your green avatar one tile at a time. "
        "Avoid traps and collect 10 yellow gems to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Navigate the grid to collect gems while "
        "dodging traps that activate in patterns. Each move matters!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Arial", 48, bold=True)

        # --- Colors ---
        self.COLOR_BG = (34, 34, 34) # #222222
        self.COLOR_GRID = (68, 68, 68) # #444444
        self.COLOR_PLAYER = (0, 255, 127) # SpringGreen
        self.COLOR_GEM = (255, 215, 0) # Gold
        self.COLOR_TRAP_1 = (255, 68, 68) # #FF4444
        self.COLOR_TRAP_2 = (204, 68, 255) # #CC44FF
        self.COLOR_TRAP_3 = (68, 120, 255) # A nice blue
        self.COLOR_TEXT = (255, 255, 255)
        self.TRAP_COLORS = {1: self.COLOR_TRAP_1, 2: self.COLOR_TRAP_2, 3: self.COLOR_TRAP_3}

        # --- Game Grid ---
        self.grid_w, self.grid_h = 16, 10
        self.tile_w, self.tile_h = 40, 20
        self.origin_x = self.screen_width // 2
        self.origin_y = self.screen_height // 2 - (self.grid_h * self.tile_h // 2) + 50

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.gems_collected = 0
        self.game_over = False
        self.win = False
        self.player_pos = [0, 0]
        self.gems = []
        self.traps = {} # {(x,y): type}
        self.active_traps = set()
        self.particles = []
        
        # Initialize state variables
        self.reset()

        # --- Validate implementation ---
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.gems_collected = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.active_traps = set()

        # Place player in the center
        self.player_pos = [self.grid_w // 2, self.grid_h // 2]
        
        # Generate gems and traps
        self.gems = []
        self.traps = {}
        
        possible_coords = [(x, y) for x in range(self.grid_w) for y in range(self.grid_h)]
        possible_coords.remove(tuple(self.player_pos))
        self.np_random.shuffle(possible_coords)

        # Place 15 gems
        num_gems = 15
        for _ in range(num_gems):
            if possible_coords:
                self.gems.append(list(possible_coords.pop(0)))

        # Place 20 traps
        num_traps = 20
        for _ in range(num_traps):
            if possible_coords:
                pos = possible_coords.pop(0)
                trap_type = self.np_random.integers(1, 4) # type 1, 2, or 3
                self.traps[pos] = trap_type

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over or self.win:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]
        # space_held and shift_held are ignored as per the brief
        
        reward = -0.2 # Cost for taking a step

        # --- 1. Update Player Position ---
        dx, dy = 0, 0
        if movement == 1: # Up
            dy = -1
        elif movement == 2: # Down
            dy = 1
        elif movement == 3: # Left
            dx = -1
        elif movement == 4: # Right
            dx = 1
        
        if dx != 0 or dy != 0:
            target_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
            if 0 <= target_pos[0] < self.grid_w and 0 <= target_pos[1] < self.grid_h:
                self.player_pos = target_pos
        
        self.steps += 1
        
        # --- 2. Update Game State (Traps) ---
        self.active_traps.clear()
        for pos, trap_type in self.traps.items():
            if trap_type == 1 and self.steps % 3 == 0:
                self.active_traps.add(pos)
            elif trap_type == 2 and self.steps % 5 == 0:
                self.active_traps.add(pos)
            elif trap_type == 3: # Always active
                self.active_traps.add(pos)
        
        # --- 3. Check for Events & Calculate Reward ---
        player_pos_tuple = tuple(self.player_pos)
        
        # Check for gem collection
        if self.player_pos in self.gems:
            self.gems.remove(self.player_pos)
            self.gems_collected += 1
            reward = 1.0
            self.score += 1.0
            iso_pos = self._grid_to_iso(*self.player_pos)
            self._create_particles(iso_pos, self.COLOR_GEM, 30)
            # sfx: gem collect sound

        # Check for trap collision
        if player_pos_tuple in self.active_traps:
            self.game_over = True
            reward = -50.0
            self.score -= 50.0
            iso_pos = self._grid_to_iso(*self.player_pos)
            self._create_particles(iso_pos, self.TRAP_COLORS[self.traps[player_pos_tuple]], 50, 'shockwave')
            # sfx: player death/trap explosion sound

        # --- 4. Check for Termination ---
        terminated = False
        if self.game_over:
            terminated = True
        
        if self.gems_collected >= 10:
            self.win = True
            terminated = True
            reward += 10.0 # Win bonus
            self.score += 10.0
            iso_pos = self._grid_to_iso(*self.player_pos)
            self._create_particles(iso_pos, self.COLOR_PLAYER, 100, 'fountain')
            # sfx: win fanfare
            
        if self.steps >= 1000:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
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
            "gems_collected": self.gems_collected,
        }

    # --- Helper & Rendering Methods ---

    def _grid_to_iso(self, gx, gy):
        sx = self.origin_x + (gx - gy) * (self.tile_w / 2)
        sy = self.origin_y + (gx + gy) * (self.tile_h / 2)
        return int(sx), int(sy)

    def _draw_iso_tile(self, surface, color, gx, gy):
        points = [
            self._grid_to_iso(gx, gy),
            self._grid_to_iso(gx + 1, gy),
            self._grid_to_iso(gx + 1, gy + 1),
            self._grid_to_iso(gx, gy + 1)
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _create_particles(self, pos, color, count, style='burst'):
        for _ in range(count):
            if style == 'burst':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                life = self.np_random.integers(20, 40)
            elif style == 'shockwave':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(4, 6)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                life = self.np_random.integers(30, 50)
            elif style == 'fountain':
                angle = self.np_random.uniform(-math.pi * 0.8, -math.pi * 0.2)
                speed = self.np_random.uniform(2, 6)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                life = self.np_random.integers(40, 70)
            
            self.particles.append({'x': pos[0], 'y': pos[1], 'vx': vx, 'vy': vy, 'life': life, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if 'vy' in p and p['vy'] < 5: # Apply gravity to fountain particles
                 p['vy'] += 0.1 
            
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                radius = int(p['life'] / 10) + 1
                pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), max(1, radius))

    def _render_game(self):
        # Draw grid lines
        for y in range(self.grid_h + 1):
            start = self._grid_to_iso(0, y)
            end = self._grid_to_iso(self.grid_w, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for x in range(self.grid_w + 1):
            start = self._grid_to_iso(x, 0)
            end = self._grid_to_iso(x, self.grid_h)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Draw traps
        for pos, trap_type in self.traps.items():
            color = self.TRAP_COLORS[trap_type]
            if pos in self.active_traps:
                # Active trap glow effect
                glow_color = (min(255, color[0] + 70), min(255, color[1] + 70), min(255, color[2] + 70))
                self._draw_iso_tile(self.screen, glow_color, *pos)
            else:
                self._draw_iso_tile(self.screen, color, *pos)

        # Draw gems
        for gem_pos in self.gems:
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.005)) * 10
            color = (min(255, self.COLOR_GEM[0] + int(pulse*2)), min(255, self.COLOR_GEM[1] + int(pulse*1.5)), self.COLOR_GEM[2])
            
            self._draw_iso_tile(self.screen, color, *gem_pos)

        # Draw player
        if not self.game_over:
            # Player glow
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.01)) * 15
            glow_color = (self.COLOR_PLAYER[0], self.COLOR_PLAYER[1], self.COLOR_PLAYER[2], 50 + int(pulse))
            glow_surf = pygame.Surface((self.tile_w*1.5, self.tile_h*1.5), pygame.SRCALPHA)
            pygame.gfxdraw.filled_polygon(glow_surf, [(self.tile_w*0.75, 0), (self.tile_w*1.5, self.tile_h*0.75), (self.tile_w*0.75, self.tile_h*1.5), (0, self.tile_h*0.75)], glow_color)
            iso_pos = self._grid_to_iso(*self.player_pos)
            self.screen.blit(glow_surf, (iso_pos[0] - self.tile_w*0.75, iso_pos[1] - self.tile_h*0.75))
            
            # Draw main player tile
            self._draw_iso_tile(self.screen, self.COLOR_PLAYER, *self.player_pos)

        self._update_and_draw_particles()

    def _render_ui(self):
        # UI background
        ui_bg_rect = pygame.Rect(0, 0, self.screen_width, 40)
        ui_bg_surf = pygame.Surface(ui_bg_rect.size, pygame.SRCALPHA)
        ui_bg_surf.fill((0, 0, 0, 128))
        self.screen.blit(ui_bg_surf, (0, 0))

        # Score/Gems display
        gem_text = f"GEMS: {self.gems_collected} / 10"
        text_surf = self.font_ui.render(gem_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 8))
        
        # Steps display
        step_text = f"STEPS: {self.steps}"
        step_surf = self.font_ui.render(step_text, True, self.COLOR_TEXT)
        step_rect = step_surf.get_rect(topright=(self.screen_width - 10, 8))
        self.screen.blit(step_surf, step_rect)

        # Game Over / Win Message
        if self.game_over or self.win:
            msg_text = "YOU WIN!" if self.win else "GAME OVER"
            msg_color = self.COLOR_PLAYER if self.win else self.COLOR_TRAP_1
            
            msg_surf = self.font_msg.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            
            # Text shadow
            shadow_surf = self.font_msg.render(msg_text, True, (0,0,0))
            self.screen.blit(shadow_surf, (msg_rect.x + 3, msg_rect.y + 3))

            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Isometric Grid Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        action_to_take = None # We only step on a key press
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if terminated: # If game over, any key press resets
                    obs, info = env.reset()
                    terminated = False
                    continue

                action = [0, 0, 0] # Default action is no-op
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    action[0] = 1
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    action[0] = 2
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    action[0] = 4
                elif event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                    terminated = False
                    continue
                else:
                    # Don't step if a non-movement key is pressed
                    continue

                action_to_take = action
        
        if action_to_take and not terminated:
            obs, reward, terminated, truncated, info = env.step(action_to_take)
            print(f"Action: {action_to_take}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        # Render the observation from the environment to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()