
# Generated: 2025-08-28T01:30:40.192535
# Source Brief: brief_04128.md
# Brief Index: 4128

        
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
        "Controls: Arrow keys to move. Hold space to fire. Press shift to activate a temporary shield."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of descending aliens in a retro side-scrolling space shooter for 60 seconds."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    TARGET_FPS = 60
    MAX_STEPS = TARGET_FPS * 60  # 60 seconds

    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_SHIELD = (100, 150, 255, 100)
    COLOR_BULLET_PLAYER = (255, 255, 255)
    COLOR_BULLET_ENEMY = (255, 0, 255)
    COLOR_ENEMY_1 = (255, 80, 80) # Red - Linear
    COLOR_ENEMY_2 = (255, 165, 0) # Orange - Sinusoidal
    COLOR_ENEMY_3 = (255, 255, 0) # Yellow - Diagonal
    COLOR_TEXT = (255, 255, 255)
    COLOR_PARTICLE = (255, 220, 150)

    # Game parameters
    PLAYER_SPEED = 5
    BULLET_SPEED = 8
    ENEMY_BASE_SPEED = 1.5
    SHIELD_DURATION = 50
    SHIELD_COOLDOWN = 200
    PLAYER_SHOOT_COOLDOWN = 8
    NUM_STARS = 100
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.player = None
        self.enemies = []
        self.player_bullets = []
        self.enemy_bullets = []
        self.particles = []
        self.stars = []
        self.last_shift_state = 0
        
        # Initialize game state
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player = self._Player(self.np_random)
        self.enemies.clear()
        self.player_bullets.clear()
        self.enemy_bullets.clear()
        self.particles.clear()
        
        # Difficulty parameters
        self.difficulty_level = 0
        self.enemy_spawn_rates = [60, 50, 42, 35, 30, 25] # Frames per spawn
        self.enemy_spawn_timer = self.enemy_spawn_rates[0]
        self.enemy_fire_cooldowns = [120, 100, 85, 70, 60, 50] # Base cooldown

        # Background
        if not self.stars:
            for _ in range(self.NUM_STARS):
                self.stars.append(self._Star(self.np_random))

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle Input & Update Player
        self._handle_input(action)
        self.player.update()
        reward += 0.01 # Base survival reward (re-scaled from 0.1 to keep total lower)
        
        # Penalty for staying at the bottom
        if self.player.y > self.SCREEN_HEIGHT * 0.8:
            reward -= 0.02 # Re-scaled from 0.2

        # 2. Update Game Objects
        self._update_bullets()
        reward += self._update_enemies()
        self._update_particles()
        for star in self.stars:
            star.update(self.SCREEN_HEIGHT)
            
        # 3. Spawning
        self._spawn_enemies()

        # 4. Collision Detection
        hit_reward, destroyed_reward = self._handle_collisions()
        reward += hit_reward + destroyed_reward

        # 5. Update Game State
        self.steps += 1
        self._update_difficulty()
        
        # 6. Check Termination Conditions
        terminated = False
        if self.player.health <= 0:
            self._create_explosion(self.player.x, self.player.y, 40)
            # SFX: Player Explosion
            self.game_over = True
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            reward += 100 # Victory reward
            self.game_over = True
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        self.player.move(dx, dy, self.PLAYER_SPEED, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)

        if space_held and self.player.shoot_cooldown == 0:
            self.player_bullets.append(self._Bullet(self.player.x, self.player.y - 20, -self.BULLET_SPEED, True))
            self.player.shoot_cooldown = self.PLAYER_SHOOT_COOLDOWN
            # SFX: Player shoot

        if shift_held and not self.last_shift_state: # On key press
            if self.player.activate_shield(self.SHIELD_DURATION, self.SHIELD_COOLDOWN):
                # SFX: Shield activate
                pass
        self.last_shift_state = shift_held

    def _update_difficulty(self):
        new_difficulty_level = self.steps // (self.TARGET_FPS * 10) # Increase every 10s
        if new_difficulty_level > self.difficulty_level:
            self.difficulty_level = min(new_difficulty_level, len(self.enemy_spawn_rates) - 1)

    def _spawn_enemies(self):
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            spawn_x = self.np_random.uniform(20, self.SCREEN_WIDTH - 20)
            enemy_type = self.np_random.integers(0, 3)
            self.enemies.append(self._Enemy(spawn_x, -20, enemy_type, self.np_random))
            self.enemy_spawn_timer = self.enemy_spawn_rates[self.difficulty_level]

    def _update_enemies(self):
        reward = 0
        new_bullets = []
        for enemy in self.enemies[:]:
            can_shoot = enemy.update(self.ENEMY_BASE_SPEED, self.SCREEN_WIDTH)
            if can_shoot:
                cooldown = self.enemy_fire_cooldowns[self.difficulty_level]
                if self.np_random.random() < 1.0 / cooldown: # Probabilistic firing
                    # Aim at player
                    dx = self.player.x - enemy.x
                    dy = self.player.y - enemy.y
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist > 0:
                        vel_x = (dx / dist) * self.BULLET_SPEED * 0.6
                        vel_y = (dy / dist) * self.BULLET_SPEED * 0.6
                        new_bullets.append(self._Bullet(enemy.x, enemy.y, 0, False, vel_x, vel_y))
                        # SFX: Enemy shoot
            if enemy.y > self.SCREEN_HEIGHT + 20:
                self.enemies.remove(enemy)
        self.enemy_bullets.extend(new_bullets)
        return reward

    def _update_bullets(self):
        for bullet in self.player_bullets[:]:
            bullet.update()
            if bullet.y < -10:
                self.player_bullets.remove(bullet)
        for bullet in self.enemy_bullets[:]:
            bullet.update()
            if not (0 < bullet.x < self.SCREEN_WIDTH and 0 < bullet.y < self.SCREEN_HEIGHT):
                 self.enemy_bullets.remove(bullet)

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        hit_reward = 0
        destroyed_reward = 0

        # Player bullets vs Enemies
        for bullet in self.player_bullets[:]:
            for enemy in self.enemies[:]:
                if abs(bullet.x - enemy.x) < (bullet.w + enemy.w) / 2 and \
                   abs(bullet.y - enemy.y) < (bullet.h + enemy.h) / 2:
                    if bullet in self.player_bullets: self.player_bullets.remove(bullet)
                    self.enemies.remove(enemy)
                    self._create_explosion(enemy.x, enemy.y, 20)
                    # SFX: Enemy explosion
                    self.score += 10
                    destroyed_reward += 1.0
                    break
        
        # Enemy bullets vs Player
        player_rect = pygame.Rect(self.player.x - self.player.w/2, self.player.y - self.player.h/2, self.player.w, self.player.h)
        for bullet in self.enemy_bullets[:]:
            bullet_rect = pygame.Rect(bullet.x - bullet.w/2, bullet.y - bullet.h/2, bullet.w, bullet.h)
            if player_rect.colliderect(bullet_rect):
                if self.player.is_hit():
                    hit_reward -= 1.0
                    self._create_explosion(self.player.x, bullet.y, 10)
                else: # Shield blocked
                    self._create_explosion(bullet.x, bullet.y, 5, self.COLOR_SHIELD)
                self.enemy_bullets.remove(bullet)

        return hit_reward, destroyed_reward

    def _create_explosion(self, x, y, num_particles, color=None):
        for _ in range(num_particles):
            self.particles.append(self._Particle(x, y, self.np_random, color))

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for star in self.stars:
            star.draw(self.screen)
        for enemy in self.enemies:
            enemy.draw(self.screen)
        for bullet in self.enemy_bullets:
            bullet.draw(self.screen)
        if self.player.health > 0:
            self.player.draw(self.screen)
        for bullet in self.player_bullets:
            bullet.draw(self.screen)
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) // self.TARGET_FPS)
        timer_text = self.font.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Health
        for i in range(self.player.health):
            heart_points = [ (20 + i*30, 50), (10 + i*30, 40), (20 + i*30, 45), (30 + i*30, 40), (20 + i*30, 50) ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, heart_points)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player.health,
            "difficulty": self.difficulty_level
        }

    def close(self):
        pygame.quit()

    # --- Helper Classes ---
    class _Player:
        def __init__(self, rng):
            self.w, self.h = 24, 28
            self.rng = rng
            self.reset()

        def reset(self):
            self.x = GameEnv.SCREEN_WIDTH // 2
            self.y = GameEnv.SCREEN_HEIGHT - 50
            self.health = 3
            self.shield_active = False
            self.shield_timer = 0
            self.shield_cooldown = 0
            self.shoot_cooldown = 0

        def move(self, dx, dy, speed, w_bound, h_bound):
            self.x += dx * speed
            self.y += dy * speed
            self.x = np.clip(self.x, self.w // 2, w_bound - self.w // 2)
            self.y = np.clip(self.y, self.h // 2, h_bound - self.h // 2)

        def update(self):
            if self.shoot_cooldown > 0: self.shoot_cooldown -= 1
            if self.shield_cooldown > 0: self.shield_cooldown -= 1
            if self.shield_timer > 0:
                self.shield_timer -= 1
                if self.shield_timer == 0:
                    self.shield_active = False
            
        def activate_shield(self, duration, cooldown):
            if self.shield_cooldown == 0:
                self.shield_active = True
                self.shield_timer = duration
                self.shield_cooldown = cooldown
                return True
            return False

        def is_hit(self):
            if not self.shield_active:
                self.health -= 1
                return True
            return False

        def draw(self, screen):
            points = [(self.x, self.y - self.h / 2), (self.x - self.w / 2, self.y + self.h / 2), (self.x + self.w / 2, self.y + self.h / 2)]
            pygame.draw.polygon(screen, GameEnv.COLOR_PLAYER, points)
            pygame.gfxdraw.aapolygon(screen, [(int(p[0]), int(p[1])) for p in points], GameEnv.COLOR_PLAYER)
            pygame.gfxdraw.filled_circle(screen, int(self.x), int(self.y), int(self.w * 0.8), GameEnv.COLOR_PLAYER_GLOW)
            if self.shield_active:
                radius = int(max(self.w, self.h) * 0.9)
                alpha = int(100 + 100 * (self.shield_timer / GameEnv.SHIELD_DURATION))
                color = (*GameEnv.COLOR_SHIELD[:3], alpha)
                shield_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(shield_surf, radius, radius, radius, color)
                pygame.gfxdraw.aacircle(shield_surf, radius, radius, radius, (200, 220, 255, alpha))
                screen.blit(shield_surf, (int(self.x - radius), int(self.y - radius)))

    class _Enemy:
        def __init__(self, x, y, type, rng):
            self.x, self.y = x, y
            self.type = type
            self.rng = rng
            self.w, self.h = 22, 22
            
            if self.type == 1: # Sinusoidal
                self.color = GameEnv.COLOR_ENEMY_2
                self.amplitude = self.rng.uniform(30, 80)
                self.frequency = self.rng.uniform(0.01, 0.03)
                self.center_x = self.x
            elif self.type == 2: # Diagonal
                self.color = GameEnv.COLOR_ENEMY_3
                self.vx = GameEnv.ENEMY_BASE_SPEED if self.rng.random() > 0.5 else -GameEnv.ENEMY_BASE_SPEED
            else: # Linear
                self.color = GameEnv.COLOR_ENEMY_1

        def update(self, speed, w_bound):
            self.y += speed
            if self.type == 1: self.x = self.center_x + self.amplitude * math.sin(self.y * self.frequency)
            elif self.type == 2:
                self.x += self.vx
                if self.x <= self.w // 2 or self.x >= w_bound - self.w // 2: self.vx *= -1
            return True

        def draw(self, screen):
            rect = pygame.Rect(self.x - self.w/2, self.y - self.h/2, self.w, self.h)
            if self.type == 1: pygame.draw.polygon(screen, self.color, [(rect.centerx, rect.top), (rect.right, rect.centery), (rect.centerx, rect.bottom), (rect.left, rect.centery)])
            elif self.type == 2: pygame.draw.rect(screen, self.color, rect, border_radius=5)
            else: pygame.draw.rect(screen, self.color, rect)

    class _Bullet:
        def __init__(self, x, y, dy, is_player, dx=0, vel_y=None):
            self.x, self.y = x, y
            self.is_player = is_player
            self.dx = dx
            self.dy = vel_y if vel_y is not None else dy
            self.w, self.h = (3, 10) if is_player else (6, 6)
            self.color = GameEnv.COLOR_BULLET_PLAYER if is_player else GameEnv.COLOR_BULLET_ENEMY

        def update(self):
            self.x += self.dx
            self.y += self.dy

        def draw(self, screen):
            rect = pygame.Rect(self.x - self.w/2, self.y - self.h/2, self.w, self.h)
            pygame.draw.rect(screen, self.color, rect, border_radius=2)

    class _Particle:
        def __init__(self, x, y, rng, color=None):
            self.x, self.y = x, y
            self.vx = rng.uniform(-3, 3)
            self.vy = rng.uniform(-3, 3)
            self.lifespan = rng.integers(15, 30)
            self.life = self.lifespan
            self.color = color if color is not None else GameEnv.COLOR_PARTICLE
            self.radius = rng.uniform(1, 4)

        def update(self):
            self.x += self.vx; self.y += self.vy
            self.vx *= 0.95; self.vy *= 0.95
            self.life -= 1

        def draw(self, screen):
            if self.life > 0:
                alpha = int(255 * (self.life / self.lifespan))
                color = (*self.color[:3], alpha)
                temp_surf = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (self.radius, self.radius), self.radius)
                screen.blit(temp_surf, (self.x - self.radius, self.y - self.radius))

    class _Star:
        def __init__(self, rng):
            self.x = rng.uniform(0, GameEnv.SCREEN_WIDTH)
            self.y = rng.uniform(0, GameEnv.SCREEN_HEIGHT)
            self.speed = rng.uniform(0.1, 1.5)
            self.radius = max(1, self.speed * 0.8)
            self.color = (int(80 + 100 * (self.speed/1.5)),) * 3

        def update(self, h_bound):
            self.y += self.speed
            if self.y > h_bound:
                self.y = 0; self.x = random.uniform(0, GameEnv.SCREEN_WIDTH)
                
        def draw(self, screen):
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.radius))

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Space Shooter")
    clock = pygame.time.Clock()

    done = False
    total_reward = 0
    
    # Mapping from Pygame keys to action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not done:
        # --- Human Controls ---
        movement = 0 # No-op
        space = 0
        shift = 0
        
        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize first key in map
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(GameEnv.TARGET_FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()