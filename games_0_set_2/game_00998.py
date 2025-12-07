
# Generated: 2025-08-27T15:27:18.737387
# Source Brief: brief_00998.md
# Brief Index: 998

        
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


# Helper classes for game entities
class Player:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.w, self.h = 30, 20
        self.reset()

    def reset(self):
        self.pos = pygame.Vector2(50, self.screen_height / 2)
        self.speed = 8
        self.health = 100
        self.max_health = 100
        self.fire_cooldown = 0
        self.fire_rate = 5  # steps per shot

    def move(self, direction):
        if direction == "UP":
            self.pos.y -= self.speed
        elif direction == "DOWN":
            self.pos.y += self.speed
        self.pos.y = np.clip(self.pos.y, self.h, self.screen_height - self.h)

    def can_fire(self):
        return self.fire_cooldown <= 0

    def fire(self):
        self.fire_cooldown = self.fire_rate
        # sfx: player_shoot.wav
        return Projectile(self.pos + (self.w, self.h / 2), pygame.Vector2(1, 0), 20)

    def update(self):
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

    @property
    def rect(self):
        return pygame.Rect(self.pos.x, self.pos.y, self.w, self.h)

class Alien:
    def __init__(self, pos, alien_type, speed_mult, fire_rate_mult, np_random):
        self.pos = pygame.Vector2(pos)
        self.type = alien_type
        self.np_random = np_random
        
        if self.type == "grunt":
            self.w, self.h = 25, 25
            self.health = 1
            self.base_speed = 2
            self.fire_rate = int(40 / fire_rate_mult)
            self.color = (255, 80, 80) # Red
        elif self.type == "elite":
            self.w, self.h = 30, 30
            self.health = 3
            self.base_speed = 3
            self.fire_rate = int(25 / fire_rate_mult)
            self.color = (200, 80, 255) # Purple

        self.speed = self.base_speed * speed_mult
        self.fire_cooldown = self.np_random.integers(0, self.fire_rate)
        self.amplitude = self.np_random.uniform(1, 3)
        self.frequency = self.np_random.uniform(0.02, 0.05)
        self.initial_y = self.pos.y

    def update(self, player_pos):
        self.pos.x -= self.speed
        self.pos.y = self.initial_y + self.amplitude * math.sin(self.frequency * self.pos.x)
        
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
    
    def can_fire(self):
        return self.fire_cooldown <= 0

    def fire(self):
        self.fire_cooldown = self.fire_rate
        # sfx: enemy_shoot.wav
        return Projectile(self.pos + (0, self.h/2), pygame.Vector2(-1, 0), 10)

    @property
    def rect(self):
        return pygame.Rect(self.pos.x, self.pos.y, self.w, self.h)

class Projectile:
    def __init__(self, pos, direction, speed):
        self.pos = pygame.Vector2(pos)
        self.direction = direction.normalize()
        self.speed = speed
        self.size = 10 if self.direction.x > 0 else 8

    def update(self):
        self.pos += self.direction * self.speed

    @property
    def rect(self):
        return pygame.Rect(self.pos.x, self.pos.y, self.size, self.size / 2)

class Particle:
    def __init__(self, pos, np_random):
        self.pos = pygame.Vector2(pos)
        self.np_random = np_random
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 5)
        self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.lifespan = self.np_random.integers(10, 20)
        self.color = random.choice([(255, 255, 255), (255, 200, 0), (255, 100, 0)])
        self.size = self.np_random.uniform(2, 5)

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95
        self.lifespan -= 1
        self.size = max(0, self.size - 0.2)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓ to move. Press space to fire your weapon."
    )
    game_description = (
        "Survive waves of descending alien attackers for 3 minutes in this retro side-scrolling shooter."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 10
    MAX_TIME_S = 180
    
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_PROJECTILE = (100, 200, 255)
    COLOR_ENEMY_PROJECTILE = (255, 255, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR = (0, 200, 100)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 64)

        self.player = Player(self.WIDTH, self.HEIGHT)
        
        self.reset()
        self.validate_implementation()
    
    def _initialize_stars(self):
        self.stars = []
        for _ in range(150):
            layer = self.np_random.integers(1, 4)
            self.stars.append({
                "pos": pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                "speed": layer * 0.5,
                "size": layer,
                "color": (50 * layer, 50 * layer, 60 * layer)
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player.reset()
        
        self.steps = 0
        self.score = 0
        self.timer = self.MAX_TIME_S
        self.game_over = False
        self.game_won = False
        
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        
        self._initialize_stars()

        # Difficulty progression state
        self.spawn_timer = 0
        self.base_spawn_rate = 20 # steps
        self.alien_speed_multiplier = 1.0
        self.alien_fire_rate_multiplier = 1.0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.1 # Survival reward
        
        # --- 1. Handle Input ---
        if not self.game_over:
            if movement == 1: self.player.move("UP")
            elif movement == 2: self.player.move("DOWN")

            if space_held and self.player.can_fire():
                self.player_projectiles.append(self.player.fire())

        # --- 2. Update Game State ---
        self.steps += 1
        if not self.game_over:
            self.timer = max(0, self.timer - 1.0 / self.FPS)

        self.player.update()
        
        # Update entities
        for proj in self.player_projectiles: proj.update()
        for proj in self.alien_projectiles: proj.update()
        for alien in self.aliens: 
            alien.update(self.player.pos)
            if alien.can_fire() and alien.pos.x < self.WIDTH:
                self.alien_projectiles.append(alien.fire())
        for particle in self.particles: particle.update()
        for star in self.stars:
            star["pos"].x -= star["speed"]
            if star["pos"].x < 0:
                star["pos"].x = self.WIDTH
                star["pos"].y = self.np_random.uniform(0, self.HEIGHT)
        
        # --- 3. Collision Detection & Rewards ---
        if not self.game_over:
            # Player projectiles vs Aliens
            for proj in self.player_projectiles[:]:
                for alien in self.aliens[:]:
                    if proj.rect.colliderect(alien.rect):
                        alien.health -= 1
                        if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                        if alien.health <= 0:
                            # sfx: explosion.wav
                            self._create_explosion(alien.pos)
                            self.aliens.remove(alien)
                            reward += 1.0
                            self.score += 100
                        break

            # Alien projectiles vs Player
            for proj in self.alien_projectiles[:]:
                if proj.rect.colliderect(self.player.rect):
                    self.player.health -= 10
                    # sfx: player_hit.wav
                    self._create_explosion(proj.pos, num=5, color=(255,255,255))
                    self.alien_projectiles.remove(proj)
                    break
        
        # --- 4. Spawning & Difficulty Scaling ---
        self._update_difficulty()
        self.spawn_timer -= 1
        if self.spawn_timer <= 0 and not self.game_over:
            self._spawn_alien()
            self.spawn_timer = self.base_spawn_rate

        # --- 5. Cleanup ---
        self.player_projectiles = [p for p in self.player_projectiles if p.pos.x < self.WIDTH]
        self.alien_projectiles = [p for p in self.alien_projectiles if p.pos.x > 0]
        self.aliens = [a for a in self.aliens if a.pos.x > -a.w]
        self.particles = [p for p in self.particles if p.lifespan > 0]

        # --- 6. Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.game_won:
                reward += 100.0
            else: # Lost
                reward -= 100.0
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _update_difficulty(self):
        time_elapsed = self.MAX_TIME_S - self.timer
        if time_elapsed > 120: # Phase 3
            self.base_spawn_rate = 10
            self.alien_speed_multiplier = 1.1
            self.alien_fire_rate_multiplier = 1.1
        elif time_elapsed > 60: # Phase 2
            self.base_spawn_rate = 15
            self.alien_speed_multiplier = 1.05
            self.alien_fire_rate_multiplier = 1.05

    def _spawn_alien(self):
        y_pos = self.np_random.uniform(30, self.HEIGHT - 30)
        pos = (self.WIDTH + 20, y_pos)
        
        time_elapsed = self.MAX_TIME_S - self.timer
        alien_type = "grunt"
        if time_elapsed > 60 and self.np_random.random() > 0.6:
            alien_type = "elite"
            
        self.aliens.append(Alien(pos, alien_type, self.alien_speed_multiplier, self.alien_fire_rate_multiplier, self.np_random))

    def _check_termination(self):
        if self.game_over:
            return True
        if self.player.health <= 0:
            self.game_over = True
            # sfx: game_over.wav
            self._create_explosion(self.player.pos, num=100)
            return True
        if self.timer <= 0:
            self.game_over = True
            self.game_won = True
            # sfx: victory.wav
            return True
        return False
        
    def _create_explosion(self, pos, num=20, color=None):
        for _ in range(num):
            p = Particle(pos, self.np_random)
            if color: p.color = color
            self.particles.append(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Stars
        for star in self.stars:
            pygame.draw.circle(self.screen, star["color"], star["pos"], star["size"])
            
        # Player
        if self.player.health > 0:
            p = self.player.pos
            w, h = self.player.w, self.player.h
            points = [(p.x, p.y), (p.x, p.y + h), (p.x + w, p.y + h / 2)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
            # Engine flame
            if not self.game_over:
                flame_len = self.np_random.uniform(5, 15)
                flame_h = h * 0.6
                flame_y = p.y + h * 0.2
                flame_points = [(p.x, flame_y), (p.x, flame_y + flame_h), (p.x - flame_len, p.y + h/2)]
                pygame.gfxdraw.aapolygon(self.screen, flame_points, (255,150,0))
                pygame.gfxdraw.filled_polygon(self.screen, flame_points, (255,200,50))

        # Aliens
        for alien in self.aliens:
            if alien.type == "grunt":
                pygame.gfxdraw.aacircle(self.screen, int(alien.pos.x + alien.w/2), int(alien.pos.y + alien.h/2), int(alien.w/2), alien.color)
                pygame.gfxdraw.filled_circle(self.screen, int(alien.pos.x + alien.w/2), int(alien.pos.y + alien.h/2), int(alien.w/2), alien.color)
            elif alien.type == "elite":
                p = alien.pos
                w, h = alien.w, alien.h
                points = [(p.x, p.y + h/2), (p.x + w/2, p.y), (p.x + w, p.y + h/2), (p.x + w/2, p.y + h)]
                pygame.gfxdraw.aapolygon(self.screen, points, alien.color)
                pygame.gfxdraw.filled_polygon(self.screen, points, alien.color)
                
        # Projectiles
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJECTILE, proj.rect, border_radius=2)
        for proj in self.alien_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_PROJECTILE, proj.rect, border_radius=2)

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p.color, p.pos, p.size)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player.health / self.player.max_health
        bar_width = 150
        bar_height = 15
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), bar_height))
        
        health_text = self.font_ui.render(f"HP: {self.player.health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 30))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(center=(self.WIDTH / 2, 20))
        self.screen.blit(score_text, score_rect)
        
        # Timer
        mins, secs = divmod(int(self.timer), 60)
        timer_text = self.font_ui.render(f"TIME: {mins:02}:{secs:02}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over / Win Message
        if self.game_over:
            if self.game_won:
                msg = "MISSION COMPLETE"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "player_health": self.player.health,
            "game_won": self.game_won
        }

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's a demonstration of the environment and not part of the required implementation
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    pygame.display.set_caption("Galactic Survival")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("      GALACTIC SURVIVAL")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
        
        if keys[pygame.K_ESCAPE]:
            running = False

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        # Control the frame rate for human play
        clock.tick(30) # Run at a smooth 30 FPS for human players
        
    pygame.quit()