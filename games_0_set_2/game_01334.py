import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move horizontally. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist, procedurally generated Alien Invaders game. Defeat descending waves of aliens."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10000
        self.INITIAL_ALIEN_COUNT = 50
        self.INITIAL_LIVES = 3

        # Action and observation space
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_PLAYER_PROJ = (255, 255, 0)
        self.COLOR_ALIEN_PROJ = (200, 0, 255)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_PARTICLE_EXP = [(255, 50, 50), (255, 150, 50), (255, 255, 255)]
        self.COLOR_PARTICLE_HIT = [(0, 255, 128), (255, 255, 255)]

        # Initialize state variables (will be set in reset)
        self.player_pos = None
        self.player_lives = None
        self.player_speed = 6.0
        self.player_fire_cooldown = 0
        self.player_fire_rate = 10 # frames between shots

        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        
        self.alien_base_speed = 1.0
        self.alien_base_fire_rate = 0.002

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Pre-generate stars for parallax effect
        self.stars = [
            (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 3))
            for _ in range(100)
        ]

        # Call validation at the end of init
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_lives = self.INITIAL_LIVES
        self.player_pos = [self.WIDTH / 2, self.HEIGHT - 40]
        self.player_fire_cooldown = 0

        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []

        self.alien_base_speed = 1.0
        self.alien_base_fire_rate = 0.002
        
        # Generate aliens in a grid
        rows = 5
        cols = self.INITIAL_ALIEN_COUNT // rows
        x_spacing = self.WIDTH * 0.8 / cols
        y_spacing = 40
        start_x = self.WIDTH * 0.1
        start_y = 50

        for row in range(rows):
            for col in range(cols):
                base_x = start_x + col * x_spacing
                pos = [base_x, start_y + row * y_spacing]
                self.aliens.append({
                    'pos': pos,
                    'base_x': base_x,
                    'pattern': 'sin' if row % 2 == 0 else 'cos',
                    'size': 8,
                    'amplitude': self.np_random.uniform(20, 50),
                    'frequency': self.np_random.uniform(0.01, 0.03)
                })

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Survival reward

        # --- ACTION HANDLING ---
        movement = action[0]
        space_held = action[1] == 1
        
        # Player Movement
        if movement == 3:  # Left
            self.player_pos[0] -= self.player_speed
        elif movement == 4:  # Right
            self.player_pos[0] += self.player_speed
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)
        
        # Player Firing
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1

        if space_held and self.player_fire_cooldown <= 0:
            self.player_projectiles.append({
                'pos': [self.player_pos[0], self.player_pos[1] - 15],
                'speed': -12.0
            })
            self.player_fire_cooldown = self.player_fire_rate
            # sound: player_shoot

        # --- GAME LOGIC UPDATE ---
        self._update_projectiles()
        reward += self._update_aliens()
        self._update_particles()
        reward += self._handle_collisions()
        self._update_difficulty()
        
        self.steps += 1
        
        # --- TERMINATION CHECK ---
        terminated = False
        if self.player_lives <= 0:
            reward -= 100
            terminated = True
        elif not self.aliens:
            reward += 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_projectiles(self):
        # Move player projectiles
        for p in self.player_projectiles:
            p['pos'][1] += p['speed']
        self.player_projectiles = [p for p in self.player_projectiles if p['pos'][1] > 0]

        # Move alien projectiles
        for p in self.alien_projectiles:
            p['pos'][1] += p['speed']
        self.alien_projectiles = [p for p in self.alien_projectiles if p['pos'][1] < self.HEIGHT]

    def _update_aliens(self):
        reward = 0
        aliens_to_remove = []
        for alien in self.aliens:
            # Movement
            t = self.steps * alien['frequency']
            if alien['pattern'] == 'sin':
                offset = alien['amplitude'] * math.sin(t)
            else: # cos
                offset = alien['amplitude'] * math.cos(t)
            alien['pos'][0] = alien['base_x'] + offset
            alien['pos'][1] += 0.05 # slow descent

            # Firing
            if self.np_random.random() < self.alien_base_fire_rate:
                self.alien_projectiles.append({
                    'pos': [alien['pos'][0], alien['pos'][1] + alien['size']],
                    'speed': 4.0
                })
                # sound: alien_shoot

            # Check if alien reached bottom
            if alien['pos'][1] > self.HEIGHT - 50:
                 self.player_lives -= 1
                 reward -= 10
                 aliens_to_remove.append(alien)
                 self._create_particles(self.player_pos, self.COLOR_PARTICLE_HIT, 30)
        
        if aliens_to_remove:
            self.aliens = [a for a in self.aliens if a not in aliens_to_remove]

        return reward

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        for p_proj in self.player_projectiles[:]:
            p_rect = pygame.Rect(p_proj['pos'][0]-2, p_proj['pos'][1]-4, 4, 8)
            for alien in self.aliens[:]:
                a_rect = pygame.Rect(alien['pos'][0]-alien['size'], alien['pos'][1]-alien['size'], alien['size']*2, alien['size']*2)
                if p_rect.colliderect(a_rect):
                    # sound: alien_explosion
                    self._create_particles(alien['pos'], self.COLOR_PARTICLE_EXP, 20)
                    self.aliens.remove(alien)
                    self.player_projectiles.remove(p_proj)
                    self.score += 1
                    reward += 1
                    break

        # Alien projectiles vs Player
        player_rect = pygame.Rect(self.player_pos[0]-15, self.player_pos[1]-7, 30, 14)
        for a_proj in self.alien_projectiles[:]:
            a_proj_rect = pygame.Rect(a_proj['pos'][0]-3, a_proj['pos'][1]-3, 6, 6)
            if player_rect.colliderect(a_proj_rect):
                # sound: player_hit
                self.alien_projectiles.remove(a_proj)
                self.player_lives -= 1
                reward -= 10
                self._create_particles(self.player_pos, self.COLOR_PARTICLE_HIT, 30)
                # Reset player position to center after hit
                self.player_pos = [self.WIDTH / 2, self.HEIGHT - 40]
                break
        
        return reward

    def _update_difficulty(self):
        if self.steps > 0:
            if self.steps % 100 == 0:
                self.alien_base_speed += 0.02
            if self.steps % 50 == 0:
                self.alien_base_fire_rate = min(0.05, self.alien_base_fire_rate + 0.001)

    def _create_particles(self, pos, colors, count):
        for _ in range(count):
            self.particles.append({
                'pos': list(pos),
                'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)],
                'life': self.np_random.integers(10, 30),
                'color': random.choice(colors),
                'radius': self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "aliens_remaining": len(self.aliens)
        }
        
    def _render_game(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        
        # Stars
        star_steps = self.steps if self.steps is not None else 0
        for x, y, z in self.stars:
            color = 50 + z * 50
            py, px = (y + star_steps // (4-z)) % self.HEIGHT, x
            pygame.draw.circle(self.screen, (color, color, color), (px, py), z-1 if z > 1 else 1)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = p['color']
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), (*color, alpha))

        # Alien Projectiles
        for p in self.alien_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.rect(self.screen, self.COLOR_ALIEN_PROJ, (pos[0]-3, pos[1]-3, 6, 6))

        # Player Projectiles
        for p in self.player_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, (pos[0]-2, pos[1]-6, 4, 12))

        # Aliens
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            s = alien['size']
            points = [(x, y - s), (x - s, y + s//2), (x + s, y + s//2)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ALIEN)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ALIEN)
            
        # Player Ship
        if self.player_lives is not None and self.player_lives > 0:
            x, y = int(self.player_pos[0]), int(self.player_pos[1])
            w, h = 30, 14
            # Glow effect
            glow_surf = pygame.Surface((w * 2, h * 2), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, (0, 0, w*2, h*2), border_radius=8)
            self.screen.blit(glow_surf, (x - w, y - h//2 - h//2))
            # Main ship body
            player_rect = (x - w//2, y - h//2, w, h)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
            # Cockpit
            pygame.draw.rect(self.screen, self.COLOR_BG, (x-5, y-3, 10, 6), border_radius=2)
            
        self._render_ui()

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score or 0}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 10))

        # Lives
        if self.player_lives is not None:
            for i in range(self.player_lives):
                self._draw_heart(25 + i * 35, 25, 12)
            
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if not self.aliens:
                msg = "VICTORY"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_ALIEN
            
            end_text = self.font.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _draw_heart(self, x, y, size):
        points = [
            (x, y - size // 4),
            (x - size // 2, y - size // 2),
            (x - size // 2, y),
            (x, y + size // 2),
            (x + size // 2, y),
            (x + size // 2, y - size // 2),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ALIEN)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ALIEN)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # FIX: Test reset first to initialize the game state before testing observation.
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        assert info['aliens_remaining'] == self.INITIAL_ALIEN_COUNT
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test reward logic (simple case)
        self.reset()
        self.aliens = [{'pos': [self.WIDTH/2, 50], 'size': 10, 'base_x': self.WIDTH/2, 'pattern': 'sin', 'amplitude': 0, 'frequency': 0}] # one alien
        self.player_projectiles = [{'pos': [self.WIDTH/2, 60], 'speed': -10}] # one projectile about to hit
        reward = self._handle_collisions()
        assert reward == 1
        assert self.score == 1
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Play Example ---
    # This part requires a display. If running in a headless environment, this will fail.
    try:
        import os
        if os.environ.get("SDL_VIDEODRIVER", "") == "dummy":
             raise ImportError("Cannot run display in dummy mode.")
             
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Alien Invaders Gym Env")
        
        obs, info = env.reset()
        terminated = False
        
        # Main game loop for human play
        while not terminated:
            # Action defaults
            movement = 0 # no-op
            space_held = 0
            shift_held = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            # Get key presses
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_held = 1

            action = [movement, space_held, shift_held]
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the observation to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Control frame rate
            env.clock.tick(30)
            
    except (ImportError, pygame.error) as e:
        print(f"Skipping human play example: {e}")
        # --- Agent Interaction Example (headless) ---
        print("\nRunning headless agent interaction example...")
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        step_count = 0
        while not terminated:
            action = env.action_space.sample()  # Random agent
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            if terminated or truncated:
                print(f"Episode finished after {step_count} steps.")
                print(f"Final score: {info['score']}, Total reward: {total_reward:.2f}")
                print(f"Reason: {'Victory' if not info['aliens_remaining'] else 'Defeat'}")
                break

    env.close()