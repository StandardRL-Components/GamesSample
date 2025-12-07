
# Generated: 2025-08-27T15:22:07.828062
# Source Brief: brief_00968.md
# Brief Index: 968

        
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


# Helper function for linear interpolation
def lerp(a, b, t):
    """Linearly interpolates between a and b by t."""
    return a + (b - a) * t

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing descriptions ---
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your square. "
        "Collect all the gems before the timer runs out!"
    )
    game_description = (
        "A fast-paced puzzle game. Navigate the grid to collect all 50 gems "
        "as quickly as possible for a high score."
    )

    # --- Game Configuration ---
    auto_advance = True
    FPS = 30
    GAME_DURATION_SECONDS = 90

    # Grid and Cell dimensions
    GRID_WIDTH, GRID_HEIGHT = 20, 10
    CELL_SIZE = 32
    GRID_PIXEL_WIDTH = GRID_WIDTH * CELL_SIZE
    GRID_PIXEL_HEIGHT = GRID_HEIGHT * CELL_SIZE
    MARGIN_X = (640 - GRID_PIXEL_WIDTH) // 2
    MARGIN_Y = (400 - GRID_PIXEL_HEIGHT) // 2

    # Player settings
    PLAYER_MOVE_COOLDOWN = 5  # Frames between moves
    PLAYER_INTERP_SPEED = 0.4 # Smoothing factor for visual movement

    # Gem settings
    TOTAL_GEMS = 50
    SPECIAL_GEMS = 5

    # Colors (Catppuccin-inspired palette)
    COLOR_BG = (30, 30, 46)
    COLOR_GRID = (76, 80, 106)
    COLOR_PLAYER = (243, 139, 168)
    COLOR_PLAYER_GLOW = (243, 139, 168, 100)
    COLOR_GEM_NORMAL = (137, 180, 250)
    COLOR_GEM_SPECIAL = (249, 226, 175)
    COLOR_UI_TEXT = (205, 214, 244)
    COLOR_TIMER_WARN = (250, 179, 135)
    COLOR_TIMER_CRIT = (243, 139, 168)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_small = pygame.font.SysFont("monospace", 16)
            self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
            self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_medium = pygame.font.Font(None, 30)
            self.font_large = pygame.font.Font(None, 60)

        self.game_over_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        self.game_over_surface.fill((0, 0, 0, 180))

        # Initialize state variables
        self.player_grid_pos = [0, 0]
        self.player_pixel_pos = [0.0, 0.0]
        self.gems = []
        self.particles = []
        self.text_popups = []
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.GAME_DURATION_SECONDS
        
        # Player state
        self.player_grid_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.player_pixel_pos = self._grid_to_pixel(self.player_grid_pos)
        self.move_cooldown = 0
        
        # Game elements
        self._generate_gems()
        self.gems_collected_count = 0
        self.last_gem_collection_step = -999

        # Effects
        self.particles = []
        self.text_popups = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Time penalty
        
        self._update_timers_and_effects()
        
        collection_info = self._handle_player_movement(action)
        if collection_info:
            reward += collection_info['value']
            self.score += collection_info['value']
            # Bonus for quick collection (within 3 moves)
            if self.steps - self.last_gem_collection_step <= 3 * self.PLAYER_MOVE_COOLDOWN:
                bonus = 5
                reward += bonus
                self.score += bonus
                self._create_text_popup(f"BONUS +{bonus}", collection_info['pos'], self.COLOR_GEM_SPECIAL)
            self.last_gem_collection_step = self.steps
        
        terminated = self._check_termination()
        
        if terminated:
            self.game_over = True
            if self.gems_collected_count == self.TOTAL_GEMS:
                reward = 50  # Win bonus
                self.score += reward # Add final bonus to score
            else: # Time ran out
                reward = -10 # Loss penalty
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_observation(self):
        self._render_background()
        self._render_gems()
        self._render_player()
        self._render_effects()
        self._render_ui()
        
        if self.game_over:
            self.screen.blit(self.game_over_surface, (0, 0))
            final_text = "YOU WIN!" if self.gems_collected_count == self.TOTAL_GEMS else "TIME'S UP!"
            text_surf = self.font_large.render(final_text, True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=(320, 180))
            self.screen.blit(text_surf, text_rect)
            
            score_surf = self.font_medium.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            score_rect = score_surf.get_rect(center=(320, 230))
            self.screen.blit(score_surf, score_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "gems_remaining": self.TOTAL_GEMS - self.gems_collected_count
        }

    # --- Internal Logic ---

    def _update_timers_and_effects(self):
        self.steps += 1
        self.time_remaining = max(0, self.time_remaining - 1.0 / self.FPS)
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
            
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Update text popups
        for t in self.text_popups[:]:
            t['pos'][1] -= 0.5 # Move up
            t['life'] -= 1
            if t['life'] <= 0:
                self.text_popups.remove(t)

    def _handle_player_movement(self, action):
        movement_action = action[0]
        if movement_action == 0 or self.move_cooldown > 0:
            return None

        dx, dy = 0, 0
        if movement_action == 1: dy = -1  # Up
        elif movement_action == 2: dy = 1   # Down
        elif movement_action == 3: dx = -1  # Left
        elif movement_action == 4: dx = 1   # Right

        new_pos = [self.player_grid_pos[0] + dx, self.player_grid_pos[1] + dy]

        if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
            self.player_grid_pos = new_pos
            self.move_cooldown = self.PLAYER_MOVE_COOLDOWN
            # Check for gem collection
            return self._check_gem_collection()
        
        return None

    def _check_gem_collection(self):
        for gem in self.gems[:]:
            if gem['grid_pos'] == self.player_grid_pos:
                self.gems.remove(gem)
                self.gems_collected_count += 1
                
                pixel_pos = self._grid_to_pixel(gem['grid_pos'])
                self._create_particle_burst(pixel_pos, gem['color'])
                self._create_text_popup(f"+{gem['value']}", pixel_pos, gem['color'])
                # Placeholder for sound effect:
                # pygame.mixer.Sound.play(self.gem_sound)
                return {'value': gem['value'], 'pos': pixel_pos}
        return None

    def _check_termination(self):
        return self.gems_collected_count == self.TOTAL_GEMS or self.time_remaining <= 0

    def _generate_gems(self):
        self.gems = []
        all_positions = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        all_positions.remove(tuple(self.player_grid_pos)) # Don't spawn gem on player

        chosen_indices = self.np_random.choice(len(all_positions), self.TOTAL_GEMS, replace=False)
        chosen_positions = [all_positions[i] for i in chosen_indices]

        for i, pos in enumerate(chosen_positions):
            is_special = i < self.SPECIAL_GEMS
            self.gems.append({
                'grid_pos': list(pos),
                'value': 5 if is_special else 1,
                'color': self.COLOR_GEM_SPECIAL if is_special else self.COLOR_GEM_NORMAL,
                'is_special': is_special,
                'anim_offset': self.np_random.random() * 2 * math.pi
            })

    # --- Rendering ---

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.MARGIN_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.MARGIN_Y), (px, self.MARGIN_Y + self.GRID_PIXEL_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.MARGIN_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.MARGIN_X, py), (self.MARGIN_X + self.GRID_PIXEL_WIDTH, py))

    def _render_gems(self):
        for gem in self.gems:
            pos = self._grid_to_pixel(gem['grid_pos'])
            
            # Pulsing animation
            anim_phase = (self.steps * 0.1 + gem['anim_offset'])
            size_pulse = (math.sin(anim_phase) + 1) / 2 # 0 to 1
            radius = int(self.CELL_SIZE * 0.25 + size_pulse * 2)
            
            # Draw glow
            glow_radius = int(radius * 1.8)
            glow_color = gem['color']
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*glow_color, 40), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

            # Draw gem
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, gem['color'])
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, gem['color'])

    def _render_player(self):
        target_pixel_pos = self._grid_to_pixel(self.player_grid_pos)
        self.player_pixel_pos = (
            lerp(self.player_pixel_pos[0], target_pixel_pos[0], self.PLAYER_INTERP_SPEED),
            lerp(self.player_pixel_pos[1], target_pixel_pos[1], self.PLAYER_INTERP_SPEED)
        )
        
        pos = self.player_pixel_pos
        size = self.CELL_SIZE * 0.6
        rect = pygame.Rect(pos[0] - size/2, pos[1] - size/2, size, size)

        # Draw glow
        glow_size = int(size * 2)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, (0, 0, glow_size, glow_size), border_radius=int(glow_size*0.3))
        self.screen.blit(glow_surf, (rect.centerx - glow_size/2, rect.centery - glow_size/2))

        # Draw player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=5)

    def _render_effects(self):
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p['pos'][0] - p['size'], p['pos'][1] - p['size']))

        # Render text popups
        for t in self.text_popups:
            alpha = max(0, min(255, int(255 * (t['life'] / t['max_life']))))
            text_surf = self.font_small.render(t['text'], True, t['color'])
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=t['pos'])
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 8))

        # Timer
        time_int = int(self.time_remaining)
        timer_color = self.COLOR_UI_TEXT
        if time_int <= 10: timer_color = self.COLOR_TIMER_CRIT
        elif time_int <= 30: timer_color = self.COLOR_TIMER_WARN
        timer_text = self.font_medium.render(f"TIME: {time_int}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(640 - 15, 8))
        self.screen.blit(timer_text, timer_rect)
        
        # Gems Remaining
        gems_text = self.font_small.render(f"GEMS: {self.gems_collected_count} / {self.TOTAL_GEMS}", True, self.COLOR_UI_TEXT)
        gems_rect = gems_text.get_rect(center=(320, 400 - 20))
        self.screen.blit(gems_text, gems_rect)

    # --- Effect Creation ---

    def _create_particle_burst(self, pos, color):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': random.randint(2, 4)
            })

    def _create_text_popup(self, text, pos, color):
        life = 40
        self.text_popups.append({
            'pos': list(pos),
            'text': text,
            'color': color,
            'life': life,
            'max_life': life
        })

    # --- Helpers ---

    def _grid_to_pixel(self, grid_pos):
        x = self.MARGIN_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.MARGIN_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return [float(x), float(y)]

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # Action defaults
        movement = 0 # no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4
            
            # The MultiDiscrete action space requires a tuple/list of actions
            action = [movement, 0, 0] # space/shift are not used
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
        if terminated:
            # Wait a bit before resetting on game over
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

    env.close()