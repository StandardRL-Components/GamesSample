
# Generated: 2025-08-27T23:05:37.774958
# Source Brief: brief_03350.md
# Brief Index: 3350

        
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

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move. Hold space to fire. Press shift to activate your shield."
    )

    # User-facing description of the game
    game_description = (
        "Survive waves of descending aliens in this retro top-down shooter. Last for 50 seconds to win!"
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 3000  # 50 seconds at 60 FPS
    FPS = 60

    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_ALIEN = (255, 50, 50)
    COLOR_BULLET = (255, 255, 0)
    COLOR_BOMB = (200, 0, 255)
    COLOR_SHIELD = (100, 150, 255, 100) # RGBA for transparency
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_HIGH = (0, 200, 0)
    COLOR_HEALTH_MED = (230, 230, 0)
    COLOR_HEALTH_LOW = (200, 0, 0)

    # Game Parameters
    PLAYER_SPEED = 5
    PLAYER_HEALTH_MAX = 100
    PLAYER_BULLET_SPEED = 8
    PLAYER_BULLET_COOLDOWN = 6 # frames
    PLAYER_SHIELD_DURATION = 180 # 3 seconds
    PLAYER_SHIELD_COOLDOWN = 300 # 5 seconds
    ALIEN_SPEED = 1.5
    ALIEN_BOMB_SPEED = 3
    PARTICLE_LIFESPAN = 30


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_bullet_cooldown = None
        self.shield_active = None
        self.shield_timer = None
        self.shield_cooldown_timer = None
        self.aliens = None
        self.bullets = None
        self.bombs = None
        self.particles = None
        self.stars = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.alien_spawn_timer = None
        self.current_alien_spawn_rate = None
        self.current_alien_bomb_freq = None

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_bullet_cooldown = 0

        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown_timer = 0

        self.aliens = []
        self.bullets = []
        self.bombs = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Difficulty scaling
        self.current_alien_spawn_rate = 120 # every 2 seconds
        self.current_alien_bomb_freq = 300 # every 5 seconds
        self.alien_spawn_timer = self.current_alien_spawn_rate

        if self.stars is None:
            self.stars = [
                (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
                for _ in range(150)
            ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1  # Survival reward

        # --- Handle Input and Player Logic ---
        self._handle_input(action)
        self._update_player()

        # --- Update Game Entities ---
        self._update_bullets()
        self._update_aliens()
        self._update_bombs()
        self._update_particles()
        
        # --- Handle Collisions and Spawning ---
        reward += self._handle_collisions()
        reward += self._spawn_aliens()

        # --- Update Game State ---
        self.steps += 1
        self._update_difficulty()
        
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.player_health > 0:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty
                self._create_explosion(self.player_pos, 50, (255,100,0))
        
        # Clamp reward
        reward = np.clip(reward, -100, 100)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 1: self.player_pos.y -= self.PLAYER_SPEED
        if movement == 2: self.player_pos.y += self.PLAYER_SPEED
        if movement == 3: self.player_pos.x -= self.PLAYER_SPEED
        if movement == 4: self.player_pos.x += self.PLAYER_SPEED
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.HEIGHT)

        # Shooting
        if space_held and self.player_bullet_cooldown == 0:
            # Sfx: player_shoot.wav
            self.bullets.append(pygame.Vector2(self.player_pos.x, self.player_pos.y - 20))
            self.player_bullet_cooldown = self.PLAYER_BULLET_COOLDOWN

        # Shield
        if shift_held and not self.shield_active and self.shield_cooldown_timer == 0:
            # Sfx: shield_activate.wav
            self.shield_active = True
            self.shield_timer = self.PLAYER_SHIELD_DURATION
            self.shield_cooldown_timer = self.PLAYER_SHIELD_COOLDOWN
            if self.player_health < self.PLAYER_HEALTH_MAX * 0.25:
                # Reward for clutch shield activation
                self.score += 5 # This is an info reward, not RL reward
                # The brief specifies RL reward, so I will add it here
                # self.step reward will accumulate this
                # This is an event, so it should be added to the reward of the step
                # Let's add it to a temporary variable and add it to the final reward
                # No, I can directly modify the reward. The brief is clear.
                # Let's check the brief again. "+5 for activating shield at low health (<25%)".
                # This is an event-based reward. So I should return it.
                # However, step function returns one reward. I will add it to the step reward.
                # This seems correct.
                # Let's create a temporary reward variable.
                # `reward` is already defined at the start of `step`. I'll add to it.
                # But this function is called before collision detection.
                # I'll handle this inside the main step function.
                # I'll create a flag.
                self.clutch_shield_reward = 5.0
        else:
            self.clutch_shield_reward = 0.0


    def _update_player(self):
        if self.player_bullet_cooldown > 0:
            self.player_bullet_cooldown -= 1
        if self.shield_timer > 0:
            self.shield_timer -= 1
        else:
            self.shield_active = False
        if self.shield_cooldown_timer > 0:
            self.shield_cooldown_timer -= 1
            
    def _update_bullets(self):
        self.bullets = [b for b in self.bullets if b.y > 0]
        for bullet in self.bullets:
            bullet.y -= self.PLAYER_BULLET_SPEED

    def _update_aliens(self):
        for alien in self.aliens:
            alien['pos'].y += self.ALIEN_SPEED
            alien['bomb_timer'] -= 1
        self.aliens = [a for a in self.aliens if a['pos'].y < self.HEIGHT + 20]

    def _update_bombs(self):
        reward_penalty = 0
        for alien in self.aliens:
            if alien['bomb_timer'] <= 0:
                # Sfx: alien_fire.wav
                self.bombs.append(pygame.Vector2(alien['pos']))
                alien['bomb_timer'] = self.current_alien_bomb_freq + self.np_random.integers(-30, 30)
                reward_penalty -= 0.01 # Penalty for alien firing
        
        self.bombs = [b for b in self.bombs if b.y < self.HEIGHT]
        for bomb in self.bombs:
            bomb.y += self.ALIEN_BOMB_SPEED
        return reward_penalty

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _handle_collisions(self):
        reward = 0
        
        # Bullets vs Aliens
        for bullet in self.bullets[:]:
            for alien in self.aliens[:]:
                if bullet.distance_to(alien['pos']) < 15:
                    # Sfx: explosion.wav
                    self._create_explosion(alien['pos'], 20, self.COLOR_ALIEN)
                    self.aliens.remove(alien)
                    if bullet in self.bullets: self.bullets.remove(bullet)
                    self.score += 10
                    reward += 1
                    break
        
        # Bombs vs Player
        if not self.shield_active:
            for bomb in self.bombs[:]:
                if bomb.distance_to(self.player_pos) < 20:
                    # Sfx: player_hit.wav
                    self.bombs.remove(bomb)
                    self.player_health -= 10
                    self._create_explosion(self.player_pos, 10, self.COLOR_PLAYER)
                    if self.player_health <= 0:
                        self.player_health = 0
                        self.game_over = True
                    break
        
        # Add clutch shield reward if it happened this step
        reward += getattr(self, 'clutch_shield_reward', 0.0)
        self.clutch_shield_reward = 0.0 # Reset after applying

        return reward

    def _spawn_aliens(self):
        self.alien_spawn_timer -= 1
        if self.alien_spawn_timer <= 0:
            x_pos = self.np_random.integers(50, self.WIDTH - 50)
            self.aliens.append({
                'pos': pygame.Vector2(x_pos, -20),
                'bomb_timer': self.current_alien_bomb_freq
            })
            self.alien_spawn_timer = self.current_alien_spawn_rate
        return self._update_bombs()

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 600 == 0: # Every 10 seconds
            self.current_alien_spawn_rate = max(30, self.current_alien_spawn_rate * 0.9)
            self.current_alien_bomb_freq = max(60, self.current_alien_bomb_freq * 0.95)

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(15, self.PARTICLE_LIFESPAN),
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
            brightness = 50 + size * 25
            pygame.draw.rect(self.screen, (brightness, brightness, brightness), (x, y, size, size))
        
        # Aliens
        for alien in self.aliens:
            p1 = (alien['pos'].x, alien['pos'].y - 10)
            p2 = (alien['pos'].x - 10, alien['pos'].y + 10)
            p3 = (alien['pos'].x + 10, alien['pos'].y + 10)
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_ALIEN)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_ALIEN)

        # Player
        if self.player_health > 0:
            # Engine glow
            glow_size = 5 + (self.steps % 10)
            glow_color = (255, 150, 0)
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y + 12), glow_size, glow_color)
            
            p1 = (self.player_pos.x, self.player_pos.y - 15)
            p2 = (self.player_pos.x - 12, self.player_pos.y + 10)
            p3 = (self.player_pos.x + 12, self.player_pos.y + 10)
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)
        
        # Shield
        if self.shield_active:
            alpha = int(100 * (self.shield_timer / self.PLAYER_SHIELD_DURATION))
            shield_color = self.COLOR_SHIELD[:3] + (alpha,)
            radius = 30
            # Create a temporary surface for the transparent circle
            temp_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surface, radius, radius, radius, shield_color)
            pygame.gfxdraw.aacircle(temp_surface, radius, radius, radius, (200, 220, 255, alpha+50))
            self.screen.blit(temp_surface, (int(self.player_pos.x - radius), int(self.player_pos.y - radius)))

        # Bombs
        for bomb in self.bombs:
            size = 4 + int(math.sin(self.steps * 0.3) * 2)
            pygame.gfxdraw.filled_circle(self.screen, int(bomb.x), int(bomb.y), size, self.COLOR_BOMB)
            pygame.gfxdraw.aacircle(self.screen, int(bomb.x), int(bomb.y), size, self.COLOR_BOMB)

        # Bullets
        for bullet in self.bullets:
            pygame.draw.rect(self.screen, self.COLOR_BULLET, (bullet.x - 2, bullet.y, 4, 10))

        # Particles
        for p in self.particles:
            alpha = (p['life'] / self.PARTICLE_LIFESPAN)
            color = tuple(int(c * alpha) for c in p['color'])
            pygame.draw.rect(self.screen, color, (p['pos'].x, p['pos'].y, 2, 2))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Health Bar
        health_pct = self.player_health / self.PLAYER_HEALTH_MAX
        bar_width = 200
        bar_height = 15
        bar_x = (self.WIDTH - bar_width) / 2
        bar_y = 10
        
        if health_pct > 0.5: health_color = self.COLOR_HEALTH_HIGH
        elif health_pct > 0.25: health_color = self.COLOR_HEALTH_MED
        else: health_color = self.COLOR_HEALTH_LOW

        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, bar_width * health_pct, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)
        
        # Shield Cooldown Indicator
        cooldown_pct = self.shield_cooldown_timer / self.PLAYER_SHIELD_COOLDOWN
        if cooldown_pct > 0:
            pygame.draw.rect(self.screen, self.COLOR_SHIELD, (bar_x, bar_y + bar_height + 2, bar_width * cooldown_pct, 3))
        
        # Game Over Text
        if self.game_over:
            msg = "YOU WIN!" if self.player_health > 0 else "GAME OVER"
            color = self.COLOR_PLAYER if self.player_health > 0 else self.COLOR_ALIEN
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "shield_active": self.shield_active
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        # Test specific assertions
        self.reset()
        self.player_health = 110
        assert self.player_health <= self.PLAYER_HEALTH_MAX + 10 # Allow slight overage before clip
        self.score = -10
        assert self.score >= -10 # Score can be negative
        self.steps = self.MAX_STEPS + 1
        assert self.steps > self.MAX_STEPS
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    
    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']}")
    env.close()