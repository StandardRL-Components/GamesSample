
# Generated: 2025-08-28T03:05:32.189282
# Source Brief: brief_01913.md
# Brief Index: 1913

        
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
        "Controls: Arrow keys to move. Hold Shift for a temporary shield. Press Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro top-down space shooter. Survive 10 waves of aliens, using your shield strategically to score bonus points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        
        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_SHIELD = (100, 150, 255)
        self.COLOR_PLAYER_PROJ = (255, 255, 255)
        self.COLOR_ALIEN_STD = (255, 80, 80)
        self.COLOR_ALIEN_FAST = (80, 150, 255)
        self.COLOR_ALIEN_SHOOTER = (255, 255, 80)
        self.COLOR_ALIEN_PROJ = {
            0: (255, 120, 120),
            1: (120, 180, 255),
            2: (255, 255, 120)
        }
        self.COLOR_TEXT = (220, 220, 220)
        
        # Fonts
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 24)

        # Game parameters
        self.MAX_STEPS = 5000
        self.MAX_WAVES = 10
        self.PLAYER_SPEED = 5
        self.PLAYER_FIRE_COOLDOWN = 6 # frames
        self.PLAYER_PROJ_SPEED = 10
        self.ALIEN_PROJ_SPEED = 4
        self.SHIELD_DURATION = 5 # frames
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_health = 0
        self.player_projectiles = []
        self.aliens = []
        self.alien_projectiles = []
        self.particles = []
        self.current_wave = 0
        self.shield_timer = 0
        self.last_fire_step = 0
        self.last_closest_alien_dist = 0.0
        self.np_random = None
        self.player_fire_timer = 0
        self.stars = []

        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT * 0.85], dtype=np.float32)
        self.player_health = 3
        self.player_projectiles = []
        self.aliens = []
        self.alien_projectiles = []
        self.particles = []
        self.current_wave = 0
        self.shield_timer = 0
        self.last_fire_step = 0
        self.player_fire_timer = 0
        
        self.stars = [
            (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 2))
            for _ in range(100)
        ]

        self._spawn_next_wave()
        
        closest_alien = self._get_closest_alien()
        if closest_alien:
            self.last_closest_alien_dist = np.linalg.norm(self.player_pos - closest_alien['pos'])
        else:
            self.last_closest_alien_dist = 0.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        self.steps += 1
        reward = 0

        # --- Update Game Logic ---
        reward += self._update_player(movement, space_held, shift_held)
        self._update_aliens()
        self._update_projectiles()
        self._update_particles()
        
        reward += self._handle_collisions()

        # --- Wave Progression ---
        if not self.aliens and self.current_wave <= self.MAX_WAVES:
            reward += 100  # Wave clear bonus
            self._spawn_next_wave()
        
        # --- Continuous Rewards ---
        closest_alien = self._get_closest_alien()
        if closest_alien:
            dist = np.linalg.norm(self.player_pos - closest_alien['pos'])
            if dist < self.last_closest_alien_dist:
                reward += 0.1 # Moved closer
            else:
                reward -= 0.2 # Moved away or stood still
            self.last_closest_alien_dist = dist
        
        if self.steps - self.last_fire_step > 30: # 1 second without firing
             reward -= 0.02

        # --- Termination ---
        terminated = False
        if self.player_health <= 0:
            reward -= 100 # Death penalty
            self.game_over = True
            terminated = True
            self._create_explosion(self.player_pos, 100, self.COLOR_PLAYER)
        elif self.current_wave > self.MAX_WAVES:
            # Victory condition is just clearing wave 10, no new wave spawns
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement, space_held, shift_held):
        # Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], 10, self.WIDTH - 10)
        self.player_pos[1] = np.clip(self.player_pos[1], 10, self.HEIGHT - 10)

        # Shield
        if self.shield_timer > 0: self.shield_timer -= 1
        if shift_held and self.shield_timer == 0:
            self.shield_timer = self.SHIELD_DURATION
            # sfx: shield activate

        # Shooting
        if self.player_fire_timer > 0: self.player_fire_timer -= 1
        if space_held and self.player_fire_timer == 0:
            self.player_projectiles.append(self.player_pos.copy())
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN
            self.last_fire_step = self.steps
            # sfx: player shoot
        
        return 0

    def _update_aliens(self):
        wave_speed_mod = 1 + (self.current_wave - 1) * 0.05
        wave_fire_rate_mod = 1 + (self.current_wave - 1) * 0.05

        for alien in self.aliens:
            # Movement
            alien['pos'][0] += alien['vel'][0] * wave_speed_mod
            alien['pos'][1] += alien['vel'][1] * wave_speed_mod
            alien['move_timer'] += 1

            # Standard aliens: sweep horizontally
            if alien['type'] == 0:
                if alien['pos'][0] < 20 or alien['pos'][0] > self.WIDTH - 20:
                    alien['vel'][0] *= -1
            
            # Fast aliens: diagonal descent
            elif alien['type'] == 1:
                if alien['pos'][0] < 20 or alien['pos'][0] > self.WIDTH - 20:
                    alien['vel'][0] *= -1
                if alien['pos'][1] > self.HEIGHT * 0.6:
                    alien['vel'][1] = 0 # Stop descending

            # Shooter aliens: stationary sine wave
            elif alien['type'] == 2:
                alien['pos'][0] = alien['base_x'] + math.sin(self.steps / 30) * 50

            # Firing
            alien['fire_timer'] -= 1
            if alien['fire_timer'] <= 0:
                fire_chance = alien['fire_rate'] * wave_fire_rate_mod
                if self.np_random.random() < fire_chance:
                    direction = self.player_pos - alien['pos']
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction /= norm
                        self.alien_projectiles.append({
                            'pos': alien['pos'].copy(),
                            'vel': direction,
                            'type': alien['type']
                        })
                        alien['fire_timer'] = alien['fire_cooldown']
                        # sfx: alien shoot

    def _update_projectiles(self):
        self.player_projectiles = [p - [0, self.PLAYER_PROJ_SPEED] for p in self.player_projectiles if p[1] > 0]
        
        new_alien_projectiles = []
        for p in self.alien_projectiles:
            p['pos'] += p['vel'] * self.ALIEN_PROJ_SPEED
            if 0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT:
                new_alien_projectiles.append(p)
        self.alien_projectiles = new_alien_projectiles

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        remaining_aliens = []
        for alien in self.aliens:
            hit = False
            remaining_projectiles = []
            for proj in self.player_projectiles:
                if np.linalg.norm(proj - alien['pos']) < 15:
                    hit = True
                    # sfx: alien explosion
                    if self.shield_timer > 0:
                        reward += 5  # Shield kill bonus
                    else:
                        reward += alien['reward']
                    self.score += alien['reward'] * 10
                    self._create_explosion(alien['pos'], 30, alien['color'])
                    break # One projectile can only hit one alien
                else:
                    remaining_projectiles.append(proj)
            self.player_projectiles = remaining_projectiles
            if not hit:
                remaining_aliens.append(alien)
        self.aliens = remaining_aliens

        # Alien projectiles vs Player
        if self.shield_timer <= 0:
            remaining_alien_projectiles = []
            for proj in self.alien_projectiles:
                if np.linalg.norm(proj['pos'] - self.player_pos) < 12:
                    self.player_health -= 1
                    reward -= 1
                    self._create_explosion(self.player_pos, 20, self.COLOR_PLAYER)
                    # sfx: player hit
                else:
                    remaining_alien_projectiles.append(proj)
            self.alien_projectiles = remaining_alien_projectiles
        
        return reward

    def _spawn_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            return

        num_aliens = 5 + self.current_wave
        for i in range(num_aliens):
            alien_type_roll = self.np_random.random()
            if alien_type_roll < 0.5:
                alien_type = 0 # Standard
                color = self.COLOR_ALIEN_STD
                reward_val = 1
                fire_rate = 0.01
                vel = np.array([2.0, 0.1], dtype=np.float32)
            elif alien_type_roll < 0.8:
                alien_type = 1 # Fast
                color = self.COLOR_ALIEN_FAST
                reward_val = 2
                fire_rate = 0.005
                vel = np.array([3.0, 0.5], dtype=np.float32) * (1 if self.np_random.random() < 0.5 else -1)
            else:
                alien_type = 2 # Shooter
                color = self.COLOR_ALIEN_SHOOTER
                reward_val = 3
                fire_rate = 0.025
                vel = np.array([0.0, 0.0], dtype=np.float32)
            
            pos_x = self.WIDTH * (i + 1) / (num_aliens + 1)
            pos_y = 60 + self.np_random.integers(-20, 20)

            self.aliens.append({
                'pos': np.array([pos_x, pos_y], dtype=np.float32),
                'base_x': pos_x,
                'vel': vel,
                'type': alien_type,
                'color': color,
                'reward': reward_val,
                'fire_rate': fire_rate,
                'fire_cooldown': self.np_random.integers(100, 200),
                'fire_timer': self.np_random.integers(50, 150),
                'move_timer': 0
            })
    
    def _get_closest_alien(self):
        if not self.aliens:
            return None
        
        closest_alien = None
        min_dist = float('inf')
        for alien in self.aliens:
            dist = np.linalg.norm(self.player_pos - alien['pos'])
            if dist < min_dist:
                min_dist = dist
                closest_alien = alien
        return closest_alien

    def _create_explosion(self, position, count, color):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': position.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 25),
                'max_life': 25,
                'color': color
            })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            c = 40 + (x % 30)
            pygame.draw.rect(self.screen, (c,c,c), (x, y, size, size))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color']
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.circle(s, (color[0], color[1], color[2], alpha), (1, 1), 1)
            self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])))

        # Alien Projectiles
        for p in self.alien_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            color = self.COLOR_ALIEN_PROJ[p['type']]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, color)

        # Player Projectiles
        for p in self.player_projectiles:
            pos = (int(p[0]), int(p[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PLAYER_PROJ)

        # Aliens
        for alien in self.aliens:
            pos = (int(alien['pos'][0]), int(alien['pos'][1]))
            pygame.draw.rect(self.screen, alien['color'], (pos[0] - 8, pos[1] - 8, 16, 16))

        # Player
        if self.player_health > 0:
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            player_points = [(px, py - 10), (px - 8, py + 8), (px + 8, py + 8)]
            pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)
            
            # Shield effect
            if self.shield_timer > 0:
                radius = 15 + (self.SHIELD_DURATION - self.shield_timer) * 2
                alpha = int(100 * (self.shield_timer / self.SHIELD_DURATION))
                pygame.gfxdraw.aacircle(self.screen, px, py, radius, (*self.COLOR_SHIELD, alpha))
                pygame.gfxdraw.aacircle(self.screen, px, py, radius-1, (*self.COLOR_SHIELD, alpha))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_str = f"WAVE: {self.current_wave}/{self.MAX_WAVES}" if self.current_wave <= self.MAX_WAVES else "VICTORY"
        wave_text = self.font_main.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Health
        health_text = self.font_small.render("HEALTH:", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, self.HEIGHT - 30))
        for i in range(self.player_health):
            px, py = 100 + i * 25, self.HEIGHT - 22
            points = [(px, py - 6), (px - 4, py + 4), (px + 4, py + 4)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        
        if self.game_over:
            status = "GAME OVER" if self.player_health <= 0 else "YOU WIN!"
            end_text = self.font_main.render(status, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "health": self.player_health
        }

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

# Example usage:
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To run the game with keyboard controls ---
    import pygame
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Space Shooter")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Wait 2 seconds before resetting
            obs, info = env.reset()

        clock.tick(30) # 30 FPS

    env.close()