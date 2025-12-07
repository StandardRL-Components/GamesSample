import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Hold shift for a temporary shield. Press space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro top-down shooter. Destroy 5 waves of aliens while dodging their fire."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1500 # Increased for 5 waves
    TOTAL_WAVES = 5

    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_SHIELD = (100, 200, 255, 100)
    COLOR_PLAYER_PROJECTILE = (100, 255, 100)
    COLOR_ALIEN_PROJECTILE = (255, 100, 100)
    COLOR_TEXT = (255, 255, 255)
    ALIEN_COLORS = [
        (0, 255, 255),  # Wave 1: Cyan
        (255, 0, 255),  # Wave 2: Magenta
        (255, 255, 0),  # Wave 3: Yellow
        (255, 165, 0),  # Wave 4: Orange
        (255, 50, 50),   # Wave 5: Red
    ]

    # Game Parameters
    PLAYER_SPEED = 5
    PLAYER_SIZE = 12
    PLAYER_PROJECTILE_SPEED = 8
    ALIEN_PROJECTILE_SPEED = 4
    SHIELD_DURATION = 5
    SHIELD_COOLDOWN = 60 # Increased cooldown
    SHIELD_MISUSE_RADIUS = 75

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('monospace', 18, bold=True)
        self.font_title = pygame.font.SysFont('monospace', 40, bold=True)
        
        # Initialize state variables
        # self.reset() is called by the wrapper, but good practice to have it here for standalone use
        
        # self.validate_implementation() # This can be uncommented for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_won = False

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50], dtype=np.float32)
        self.player_lives = 3
        
        self.player_projectiles = []
        self.alien_projectiles = []
        self.aliens = []
        self.particles = []
        
        self.current_wave = 1
        self.shield_active_timer = 0
        self.shield_cooldown_timer = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self._create_starfield()
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Time penalty to encourage faster completion
        
        if not self.game_over:
            # Unpack factorized action
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            # --- Update Game Logic ---
            action_reward = self._handle_input(movement, space_held, shift_held)
            self._update_game_objects()
            collision_reward = self._handle_collisions()
            progression_reward = self._check_wave_completion()
            
            reward += action_reward + collision_reward + progression_reward

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_won:
            if self.player_lives <= 0:
                reward -= 100.0 # Big penalty for losing
            # No penalty for timeout
        elif terminated and self.game_won:
            reward += 100.0 # Big reward for winning

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0.0
        # Player Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

        # Player Firing (on key press)
        if space_held and not self.prev_space_held:
            # sfx: player_shoot.wav
            self.player_projectiles.append(self.player_pos.copy())
        self.prev_space_held = space_held

        # Player Shield (on key press)
        if shift_held and not self.prev_shift_held and self.shield_cooldown_timer == 0:
            # sfx: shield_activate.wav
            self.shield_active_timer = self.SHIELD_DURATION
            self.shield_cooldown_timer = self.SHIELD_COOLDOWN
            
            # Check for shield misuse
            is_projectile_near = False
            for proj_pos in self.alien_projectiles:
                if np.linalg.norm(self.player_pos - proj_pos) < self.SHIELD_MISUSE_RADIUS:
                    is_projectile_near = True
                    break
            if not is_projectile_near:
                reward -= 0.2

        self.prev_shift_held = shift_held
        return reward

    def _update_game_objects(self):
        # Shield timers
        if self.shield_active_timer > 0: self.shield_active_timer -= 1
        if self.shield_cooldown_timer > 0: self.shield_cooldown_timer -= 1

        # Projectiles
        self.player_projectiles = [p - np.array([0, self.PLAYER_PROJECTILE_SPEED]) for p in self.player_projectiles if p[1] > 0]
        self.alien_projectiles = [p + np.array([0, self.ALIEN_PROJECTILE_SPEED]) for p in self.alien_projectiles if p[1] < self.SCREEN_HEIGHT]
        
        # Aliens
        alien_speed = 1.0 + (self.current_wave - 1) * 0.5
        fire_prob = 0.005 + (self.current_wave - 1) * 0.005
        for alien in self.aliens:
            alien['pos'][1] += alien_speed
            if self.np_random.random() < fire_prob:
                # sfx: alien_shoot.wav
                self.alien_projectiles.append(alien['pos'].copy())
        self.aliens = [a for a in self.aliens if a['pos'][1] < self.SCREEN_HEIGHT + 20]

        # Particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        # Stars
        for star in self.stars:
            star[1] = (star[1] + star[2]) % self.SCREEN_HEIGHT
    
    def _handle_collisions(self):
        reward = 0.0
        
        # --- Player projectiles vs Aliens ---
        # We identify items to remove by their list index to avoid the numpy array comparison bug.
        aliens_hit_indices = set()
        player_projectiles_hit_indices = set()

        for p_idx, p_proj in enumerate(self.player_projectiles):
            for a_idx, alien in enumerate(self.aliens):
                # An alien can only be hit once per frame.
                if a_idx in aliens_hit_indices:
                    continue
                
                if np.linalg.norm(p_proj - alien['pos']) < self.PLAYER_SIZE:
                    # sfx: explosion.wav
                    self._create_explosion(alien['pos'], 20, self.ALIEN_COLORS[self.current_wave-1])
                    
                    aliens_hit_indices.add(a_idx)
                    player_projectiles_hit_indices.add(p_idx)
                    
                    reward += 1.0  # Reward for destroying an alien
                    self.score += 100
                    break  # A projectile can only hit one alien.

        # Rebuild lists, excluding the items that were hit.
        if aliens_hit_indices:
            self.aliens = [alien for i, alien in enumerate(self.aliens) if i not in aliens_hit_indices]
        if player_projectiles_hit_indices:
            self.player_projectiles = [p for i, p in enumerate(self.player_projectiles) if i not in player_projectiles_hit_indices]

        # --- Alien projectiles vs Player ---
        alien_projectile_hit_index = -1
        player_was_hit = False

        if self.shield_active_timer == 0:
            for i, a_proj in enumerate(self.alien_projectiles):
                if np.linalg.norm(a_proj - self.player_pos) < self.PLAYER_SIZE:
                    alien_projectile_hit_index = i
                    player_was_hit = True
                    break # Player is hit, stop checking for this frame.
        else: # Shield is active
            for i, a_proj in enumerate(self.alien_projectiles):
                if np.linalg.norm(a_proj - self.player_pos) < self.PLAYER_SIZE * 2: # Larger radius for shield
                    # sfx: shield_deflect.wav
                    alien_projectile_hit_index = i
                    self._create_explosion(a_proj, 5, self.COLOR_PLAYER_SHIELD[:3])
                    break # Projectile deflected, stop checking.
        
        # Process player hit
        if player_was_hit:
            # sfx: player_hit.wav
            self.player_lives -= 1
            reward -= 10.0 # Penalty for getting hit
            self._create_explosion(self.player_pos, 30, self.COLOR_PLAYER)
            if self.player_lives > 0:
                 self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50], dtype=np.float32)

        # Remove the single alien projectile that hit or was deflected
        if alien_projectile_hit_index != -1:
            self.alien_projectiles.pop(alien_projectile_hit_index)

        return reward

    def _check_wave_completion(self):
        if not self.aliens and not self.game_over and self.current_wave <= self.TOTAL_WAVES:
            # sfx: wave_clear.wav
            reward = 5.0
            self.current_wave += 1
            if self.current_wave <= self.TOTAL_WAVES:
                self._spawn_wave()
            else:
                self.game_won = True
                self.game_over = True
            return reward
        return 0.0

    def _check_termination(self):
        if self.player_lives <= 0:
            self.game_over = True
            return True
        if self.game_won:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _spawn_wave(self):
        num_aliens = 8 + self.current_wave * 2
        rows = math.ceil(num_aliens / 8)
        for i in range(num_aliens):
            row = i // 8
            col = i % 8
            x = (self.SCREEN_WIDTH / 9) * (col + 1)
            y = -30 - row * 40
            self.aliens.append({'pos': np.array([x, y], dtype=np.float32), 'type': self.current_wave})
    
    def _create_starfield(self):
        self.stars = []
        for _ in range(150):
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT)
            speed = self.np_random.random() * 1.5 + 0.5
            self.stars.append([x, y, speed])

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': pos.copy(), 'vel': velocity, 'life': life, 'max_life': life, 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, speed in self.stars:
            size = max(1, int(speed))
            color_val = max(50, int(speed * 60))
            pygame.draw.rect(self.screen, (color_val, color_val, color_val), (int(x), int(y), size, size))
        
        # Projectiles
        for p in self.player_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PLAYER_PROJECTILE, p.astype(int), 4)
        for p in self.alien_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_ALIEN_PROJECTILE, p.astype(int), 4)
            
        # Aliens
        for alien in self.aliens:
            pos = alien['pos'].astype(int)
            color = self.ALIEN_COLORS[alien['type'] - 1]
            pygame.draw.rect(self.screen, color, (pos[0] - 8, pos[1] - 8, 16, 16))
            
        # Player
        if self.player_lives > 0:
            p1 = self.player_pos + [0, -self.PLAYER_SIZE]
            p2 = self.player_pos + [-self.PLAYER_SIZE * 0.8, self.PLAYER_SIZE * 0.5]
            p3 = self.player_pos + [self.PLAYER_SIZE * 0.8, self.PLAYER_SIZE * 0.5]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3])
            
            # Shield effect
            if self.shield_active_timer > 0:
                s = pygame.Surface((self.PLAYER_SIZE * 4, self.PLAYER_SIZE * 4), pygame.SRCALPHA)
                alpha = int(255 * (self.shield_active_timer / self.SHIELD_DURATION))
                pygame.draw.circle(s, (*self.COLOR_PLAYER_SHIELD[:3], alpha), (self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2), self.PLAYER_SIZE * 2)
                self.screen.blit(s, (self.player_pos - self.PLAYER_SIZE * 2).astype(int))

        # Particles
        for p in self.particles:
            alpha = 255 * (p['life'] / p['max_life'])
            color = (*p['color'], alpha)
            s = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(s, color, (0,0,4,4))
            self.screen.blit(s, p['pos'].astype(int) - [2,2])

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_str = f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}" if self.current_wave <= self.TOTAL_WAVES else "ALL WAVES CLEARED"
        wave_text = self.font_ui.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.player_lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (10, self.SCREEN_HEIGHT - lives_text.get_height() - 10))

        # Shield Cooldown
        if self.shield_cooldown_timer > 0:
            cooldown_prop = self.shield_cooldown_timer / self.SHIELD_COOLDOWN
            bar_width = 100
            pygame.draw.rect(self.screen, (50, 50, 80), (self.SCREEN_WIDTH - bar_width - 10, self.SCREEN_HEIGHT - 25, bar_width, 15))
            pygame.draw.rect(self.screen, (150, 150, 200), (self.SCREEN_WIDTH - bar_width - 10, self.SCREEN_HEIGHT - 25, bar_width * (1-cooldown_prop), 15))
        
        # Game Over / Win Message
        if self.game_over:
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            end_text = self.font_title.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.current_wave,
        }
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space (after a reset)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # Create a window to display the game
    pygame.display.set_caption("Arcade Shooter")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                 print("Resetting environment.")
                 obs, info = env.reset()
                 terminated = False

        # --- Frame Rate ---
        env.clock.tick(30) # Match the intended FPS

    pygame.quit()