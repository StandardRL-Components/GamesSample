import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:31:28.941225
# Source Brief: brief_00466.md
# Brief Index: 466
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Defend your dreams in this match-3 puzzle game with a tower defense twist. "
        "Match tiles to build energy and automatically deploy defenses to stop waves of encroaching nightmares."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. "
        "Press space to swap the selected tile with the one to its right. "
        "Press shift to swap with the tile below."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- CRITICAL: Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 4
        self.TILE_SIZE = 50
        self.GRID_X, self.GRID_Y = 40, 80
        self.NUM_TILE_TYPES = 6
        self.MAX_STEPS = 10000
        self.MAX_ENERGY = 100
        self.NIGHTMARE_LEAK_LIMIT = 10
        self.DEFENSE_DURATION = 300 # steps
        self.DEFENSE_RADIUS = 70
        self.DEFENSE_SLOW_FACTOR = 0.4

        # --- Visuals & Colors ---
        self.COLOR_BG = (15, 10, 40)
        self.COLOR_GRID_BG = (25, 20, 50)
        self.COLOR_CURSOR = (255, 255, 0)
        self.TILE_COLORS = [
            (66, 135, 245),   # Blue
            (219, 68, 55),    # Red
            (244, 180, 0),    # Yellow
            (15, 157, 88),    # Green
            (171, 71, 188),   # Purple
            (255, 112, 67),   # Orange
        ]
        self.COLOR_NIGHTMARE = (190, 20, 20)
        self.COLOR_DEFENSE = (0, 255, 150)
        self.COLOR_ENERGY = (255, 200, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.Font(None, 28)
            self.font_small = pygame.font.Font(None, 20)
        except IOError:
            self.font_main = pygame.font.SysFont("sans", 28)
            self.font_small = pygame.font.SysFont("sans", 20)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.dream_energy = 0
        self.wave_number = 1
        self.nightmares = []
        self.defenses = []
        self.particles = []
        self.nightmares_leaked = 0
        self.last_action_state = [0, 0, 0]
        self.match_animation_timer = 0
        self.matched_tiles = set()
        self.swap_cooldown = 0

        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.dream_energy = 0
        self.wave_number = 1
        self.nightmares = []
        self.defenses = []
        self.particles = []
        self.nightmares_leaked = 0
        self.last_action_state = [0, 0, 0]
        self.match_animation_timer = 0
        self.matched_tiles = set()
        self.swap_cooldown = 0

        self._initialize_grid()
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Handle Actions ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_animations()
        reward += self._update_matches()
        reward += self._update_nightmares()
        self._update_defenses()
        reward += self._update_deployment()
        reward += self._update_waves()

        # Nightmare advance penalty
        reward -= 0.001 * len(self.nightmares)

        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        self.last_action_state = action

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_press, shift_press = action[0], action[1], action[2]

        # Cooldowns to prevent spamming actions
        if self.swap_cooldown > 0:
            self.swap_cooldown -= 1
        if self.match_animation_timer > 0:
            return

        # Cursor movement
        if movement == 1: self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE # Up
        elif movement == 2: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE # Down
        elif movement == 3: self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE # Left
        elif movement == 4: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE # Right

        # Tile Swapping (on press, not hold)
        if self.swap_cooldown == 0:
            swapped = False
            r, c = self.cursor_pos
            # Space: Swap right
            if space_press and not self.last_action_state[1]:
                c2 = (c + 1) % self.GRID_SIZE
                self.grid[r, c], self.grid[r, c2] = self.grid[r, c2], self.grid[r, c]
                swapped = True
            # Shift: Swap down
            elif shift_press and not self.last_action_state[2]:
                r2 = (r + 1) % self.GRID_SIZE
                self.grid[r, c], self.grid[r2, c] = self.grid[r2, c], self.grid[r, c]
                swapped = True
            
            if swapped:
                # Check if the swap creates a match
                matches = self._find_matches()
                if matches:
                    self.matched_tiles = matches
                    self.match_animation_timer = 15 # Start animation
                else: # Invalid swap, swap back
                    if space_press and not self.last_action_state[1]:
                        c2 = (c + 1) % self.GRID_SIZE
                        self.grid[r, c], self.grid[r, c2] = self.grid[r, c2], self.grid[r, c]
                    elif shift_press and not self.last_action_state[2]:
                        r2 = (r + 1) % self.GRID_SIZE
                        self.grid[r, c], self.grid[r2, c] = self.grid[r2, c], self.grid[r, c]
                self.swap_cooldown = 10 # Cooldown after any swap attempt

    def _update_animations(self):
        # Update particles
        for p in self.particles[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1
            if p[4] <= 0:
                self.particles.remove(p)

        # Update match animation
        if self.match_animation_timer > 0:
            self.match_animation_timer -= 1
    
    def _update_matches(self):
        # Process matches after animation finishes
        if self.match_animation_timer == 1 and self.matched_tiles:
            reward = 0
            num_matched = len(self.matched_tiles)
            
            # Grant energy and reward
            self.dream_energy = min(self.MAX_ENERGY, self.dream_energy + num_matched * 2)
            reward += num_matched * 0.1 # Sound: tile_match.wav

            # Create particles
            for r_idx, c_idx in self.matched_tiles:
                tile_color = self.TILE_COLORS[self.grid[r_idx, c_idx] % len(self.TILE_COLORS)]
                for _ in range(15):
                    angle = random.uniform(0, 2 * math.pi)
                    speed = random.uniform(1, 4)
                    px = self.GRID_X + c_idx * self.TILE_SIZE + self.TILE_SIZE / 2
                    py = self.GRID_Y + r_idx * self.TILE_SIZE + self.TILE_SIZE / 2
                    self.particles.append([px, py, math.cos(angle) * speed, math.sin(angle) * speed, random.randint(20, 40), tile_color])

            # Remove matched tiles and refill grid
            self._refill_grid()
            self.matched_tiles.clear()
            
            # Check for new chain-reaction matches
            new_matches = self._find_matches()
            if new_matches:
                self.matched_tiles = new_matches
                self.match_animation_timer = 15
            
            return reward
        return 0

    def _update_nightmares(self):
        reward = 0
        for nightmare in self.nightmares[:]:
            # Apply slow from defenses
            speed_mod = 1.0
            for defense in self.defenses:
                dist = math.hypot(nightmare['pos'][0] - defense['pos'][0], nightmare['pos'][1] - defense['pos'][1])
                if dist < self.DEFENSE_RADIUS:
                    speed_mod = self.DEFENSE_SLOW_FACTOR
                    break
            
            nightmare['pos'][1] += nightmare['speed'] * speed_mod

            # Check for reaching the bottom
            if nightmare['pos'][1] > self.HEIGHT:
                self.nightmares.remove(nightmare)
                self.nightmares_leaked += 1
                reward -= 1.0 # Sound: nightmare_leak.wav
        return reward

    def _update_defenses(self):
        for defense in self.defenses[:]:
            defense['duration'] -= 1
            if defense['duration'] <= 0:
                self.defenses.remove(defense)
        return 0

    def _update_deployment(self):
        # Automatic deployment when energy is full
        if self.dream_energy >= self.MAX_ENERGY:
            self.dream_energy = 0
            
            # Deploy at one of two portal locations
            portal_y = random.choice([150, 250])
            portal_x = 320
            self.defenses.append({
                'pos': (portal_x, portal_y),
                'duration': self.DEFENSE_DURATION,
                'radius': self.DEFENSE_RADIUS
            })
            # Sound: deploy_defense.wav
            return 1.0 # Reward for deploying
        return 0
    
    def _update_waves(self):
        # Spawn new wave every 600 steps (20 seconds at 30fps)
        if self.steps > 0 and self.steps % 600 == 0:
            self.wave_number += 1
            self._spawn_wave()
            return 10.0 # Reward for surviving a wave
        return 0

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number}

    def _check_termination(self):
        if self.nightmares_leaked >= self.NIGHTMARE_LEAK_LIMIT:
            self.game_over = True
        return self.game_over

    # --- Helper and Logic Functions ---

    def _initialize_grid(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                self.grid[r, c] = self.np_random.integers(0, self.NUM_TILE_TYPES)
        # Ensure no initial matches
        while self._find_matches():
            matches = self._find_matches()
            for r_idx, c_idx in matches:
                self.grid[r_idx, c_idx] = self.np_random.integers(0, self.NUM_TILE_TYPES)

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Horizontal
                if c < self.GRID_SIZE - 2 and self.grid[r, c] == self.grid[r, c + 1] == self.grid[r, c + 2]:
                    matches.update([(r, c), (r, c + 1), (r, c + 2)])
                # Vertical
                if r < self.GRID_SIZE - 2 and self.grid[r, c] == self.grid[r + 1, c] == self.grid[r + 2, c]:
                    matches.update([(r, c), (r + 1, c), (r + 2, c)])
        return matches

    def _refill_grid(self):
        for r_idx, c_idx in self.matched_tiles:
            self.grid[r_idx, c_idx] = -1 # Mark as empty

        for c in range(self.GRID_SIZE):
            empty_count = 0
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    self.grid[r + empty_count, c] = self.grid[r, c]
                    self.grid[r, c] = -1
            
            for r in range(empty_count):
                self.grid[r, c] = self.np_random.integers(0, self.NUM_TILE_TYPES)
        
        # In case the refill creates instant matches
        new_matches = self._find_matches()
        if new_matches:
            self.matched_tiles = new_matches
            self.match_animation_timer = 15

    def _spawn_wave(self):
        num_to_spawn = 2 + self.wave_number // 3
        speed = 0.5 + (self.wave_number // 5) * 0.05
        for _ in range(num_to_spawn):
            self.nightmares.append({
                'pos': [random.uniform(380, self.WIDTH - 40), random.uniform(-100, -20)],
                'speed': speed * random.uniform(0.8, 1.2)
            })

    # --- Rendering Functions ---

    def _render_all(self):
        self._render_background()
        self._render_defenses()
        self._render_grid_and_tiles()
        self._render_nightmares()
        self._render_particles()
        self._render_ui()

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Static stars for parallax effect
        for i in range(100):
            seed = i * 31
            x = (seed * 17) % self.WIDTH
            y = ((seed * 29) + (self.steps // 20)) % self.HEIGHT
            pygame.draw.circle(self.screen, (50, 50, 80), (x, y), 1)

    def _render_grid_and_tiles(self):
        # Grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, 
                         (self.GRID_X - 5, self.GRID_Y - 5, 
                          self.GRID_SIZE * self.TILE_SIZE + 10, 
                          self.GRID_SIZE * self.TILE_SIZE + 10), border_radius=10)

        # Tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile_type = self.grid[r, c]
                if tile_type == -1: continue

                color = self.TILE_COLORS[tile_type]
                rect = pygame.Rect(self.GRID_X + c * self.TILE_SIZE, 
                                   self.GRID_Y + r * self.TILE_SIZE, 
                                   self.TILE_SIZE, self.TILE_SIZE)
                
                # Pulsating effect
                pulse = (math.sin(self.steps * 0.1 + r + c) + 1) / 2
                size_mod = int(pulse * 3)
                
                # Match animation effect
                if (r, c) in self.matched_tiles:
                    anim_progress = self.match_animation_timer / 15.0
                    size_mod = int((1 - anim_progress) * self.TILE_SIZE / 2)
                    color = tuple(min(255, val + 100) for val in color)
                
                inflated_rect = rect.inflate(-8 - size_mod, -8 - size_mod)
                pygame.draw.rect(self.screen, color, inflated_rect, border_radius=8)

        # Cursor
        r, c = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_X + c * self.TILE_SIZE - 3, 
                                  self.GRID_Y + r * self.TILE_SIZE - 3, 
                                  self.TILE_SIZE + 6, self.TILE_SIZE + 6)
        
        glow_alpha = (math.sin(self.steps * 0.2) * 50 + 100)
        self._draw_glowing_rect(cursor_rect, self.COLOR_CURSOR, glow_alpha, 5)

    def _render_nightmares(self):
        for nightmare in self.nightmares:
            x, y = int(nightmare['pos'][0]), int(nightmare['pos'][1])
            # Trail effect
            for i in range(5):
                trail_alpha = 50 - i * 10
                trail_color = (*self.COLOR_NIGHTMARE, trail_alpha)
                trail_pos_y = y - i * 4 * nightmare['speed']
                s = pygame.Surface((20, 20), pygame.SRCALPHA)
                pygame.gfxdraw.filled_trigon(s, 10, 0, 0, 20, 20, 20, trail_color)
                self.screen.blit(s, (x - 10, trail_pos_y - 10))

            # Main body
            pygame.gfxdraw.filled_trigon(self.screen, x, y-10, x-10, y+10, x+10, y+10, self.COLOR_NIGHTMARE)
            pygame.gfxdraw.aatrigon(self.screen, x, y-10, x-10, y+10, x+10, y+10, self.COLOR_NIGHTMARE)

    def _render_defenses(self):
        for defense in self.defenses:
            pos = (int(defense['pos'][0]), int(defense['pos'][1]))
            radius = int(defense['radius'])
            
            # Pulsating glow
            pulse = (math.sin(self.steps * 0.15) + 1) / 2
            alpha = 30 + pulse * 30
            
            self._draw_glowing_circle(pos, radius, self.COLOR_DEFENSE, alpha)

    def _render_particles(self):
        for p in self.particles:
            color = (*p[5], int(p[4] * 6)) # Use lifetime for alpha
            pygame.draw.circle(self.screen, color, (int(p[0]), int(p[1])), int(p[4] / 10))

    def _render_ui(self):
        # UI Panel Background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (10, 10, 280, 60), border_radius=8)

        # Wave Text
        wave_text = self.font_main.render(f"Wave: {self.wave_number}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (20, 15))

        # Score Text
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (180, 18))

        # Energy Bar
        energy_label = self.font_small.render("Energy", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_label, (20, 45))
        
        bar_x, bar_y, bar_w, bar_h = 80, 45, 200, 15
        pygame.draw.rect(self.screen, (10, 5, 20), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        
        energy_ratio = self.dream_energy / self.MAX_ENERGY
        fill_w = int(bar_w * energy_ratio)
        if fill_w > 0:
            pygame.draw.rect(self.screen, self.COLOR_ENERGY, (bar_x, bar_y, fill_w, bar_h), border_radius=4)
        
        # Leaked Nightmares
        leaked_text = self.font_small.render(f"Leaked: {self.nightmares_leaked}/{self.NIGHTMARE_LEAK_LIMIT}", True, (255, 80, 80))
        self.screen.blit(leaked_text, (self.WIDTH - leaked_text.get_width() - 15, 15))

    def _draw_glowing_rect(self, rect, color, alpha, blur):
        surf = pygame.Surface((rect.width + blur*2, rect.height + blur*2), pygame.SRCALPHA)
        for i in range(blur, 0, -1):
            alpha_level = int(alpha * (1 - i/blur)**2)
            temp_color = (*color, alpha_level)
            pygame.draw.rect(surf, temp_color, (blur-i, blur-i, rect.width+i*2, rect.height+i*2), border_radius=10)
        pygame.draw.rect(surf, color, (blur, blur, rect.width, rect.height), 2, border_radius=8)
        self.screen.blit(surf, (rect.x - blur, rect.y - blur))

    def _draw_glowing_circle(self, pos, radius, color, base_alpha):
        surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        for i in range(radius // 2, 0, -2):
            alpha = int(base_alpha * (1.0 - (i / (radius // 2)))**1.5)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(surf, radius, radius, i, (*color, alpha))
        self.screen.blit(surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    # This block is not run by the evaluation service, but is useful for testing.
    # It requires pygame to be installed with a display driver.
    # To run, unset the SDL_VIDEODRIVER variable:
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    try:
        human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Dream Defense")
        clock = pygame.time.Clock()
        
        action = [0, 0, 0] # [movement, space, shift]
        
        while not done:
            # Construct action from key presses
            current_action = [0, 0, 0]
            keys = pygame.key.get_pressed()

            # Note: This is a simple mapping. Holding multiple keys might not behave as expected.
            if keys[pygame.K_UP]: current_action[0] = 1
            elif keys[pygame.K_DOWN]: current_action[0] = 2
            elif keys[pygame.K_LEFT]: current_action[0] = 3
            elif keys[pygame.K_RIGHT]: current_action[0] = 4
            
            if keys[pygame.K_SPACE]: current_action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: current_action[2] = 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            obs, reward, terminated, truncated, info = env.step(current_action)
            done = terminated or truncated

            # Render for human player
            # The observation is (H, W, C), but pygame surface is (W, H)
            # So we need to transpose it back for display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()

            clock.tick(30) # Run at 30 FPS
    except pygame.error as e:
        print(f"\nCould not create display. Pygame error: {e}")
        print("This can happen if you are running in a headless environment.")
        print("The environment is still functional for training, but cannot be rendered for human play.")
    finally:
        env.close()