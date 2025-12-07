
# Generated: 2025-08-27T18:44:59.022979
# Source Brief: brief_01934.md
# Brief Index: 1934

        
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


# Helper classes for game objects to keep the main class cleaner
class Star:
    def __init__(self, width, height):
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)
        self.speed = random.uniform(0.5, 2.0)
        self.size = int(self.speed)
        self.color = (int(100 * self.speed), int(100 * self.speed), int(100 * self.speed))

    def update(self, width):
        self.x -= self.speed
        if self.x < 0:
            self.x = width
            self.y = random.randint(0, 400)

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.size)

class Projectile:
    def __init__(self, pos, velocity, color, size):
        self.pos = pygame.Vector2(pos)
        self.velocity = pygame.Vector2(velocity)
        self.color = color
        self.size = size
        self.rect = pygame.Rect(self.pos.x - size/2, self.pos.y - size/2, size, size)

    def update(self):
        self.pos += self.velocity
        self.rect.center = self.pos

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

class Alien:
    def __init__(self, speed, fire_rate):
        self.pos = pygame.Vector2(660, random.randint(40, 360))
        self.speed = speed
        self.fire_rate = fire_rate
        self.radius = 12
        self.color = (255, 60, 60)
        self.health = 2
        self.amplitude = random.uniform(20, 80)
        self.frequency = random.uniform(0.01, 0.03)
        self.initial_y = self.pos.y
        self.rect = pygame.Rect(0, 0, self.radius * 2, self.radius * 2)

    def update(self):
        self.pos.x -= self.speed
        self.pos.y = self.initial_y + self.amplitude * math.sin(self.pos.x * self.frequency)
        self.rect.center = self.pos

    def draw(self, surface):
        # Main body
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
        # Cockpit
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius // 2, (255, 180, 180))

class Particle:
    def __init__(self, pos, color):
        self.pos = pygame.Vector2(pos)
        self.velocity = pygame.Vector2(random.uniform(-3, 3), random.uniform(-3, 3))
        self.lifespan = random.randint(15, 30) # frames
        self.color = color
        self.radius = self.lifespan / 5

    def update(self):
        self.pos += self.velocity
        self.lifespan -= 1
        self.radius = max(0, self.lifespan / 5)

    def draw(self, surface):
        if self.lifespan > 0:
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), self.color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑/↓ to move. Hold Space to fire. Dodge enemy ships and projectiles."
    )

    game_description = (
        "Pilot a spaceship, dodging enemy fire and blasting alien invaders in a retro side-scrolling shooter."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (50, 255, 50, 50)
        self.COLOR_PLAYER_PROJECTILE = (200, 255, 200)
        self.COLOR_ALIEN_PROJECTILE = (255, 200, 200)
        self.COLOR_EXPLOSION_PLAYER = (255, 255, 100)
        self.COLOR_EXPLOSION_ALIEN = (255, 150, 50)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_HEART = (255, 0, 0)
        
        # Game constants
        self.PLAYER_SPEED = 6
        self.PLAYER_RADIUS = 15
        self.PLAYER_FIRE_COOLDOWN_MAX = 8 # frames
        self.PLAYER_IFRAME_DURATION = 90 # frames
        self.TOTAL_ALIENS = 50
        self.MAX_STEPS = 30 * 60 # 60 seconds at 30fps

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = pygame.Vector2(100, self.HEIGHT // 2)
        self.player_lives = 3
        self.player_iframes = 0
        self.player_fire_cooldown = 0
        self.player_rect = pygame.Rect(0, 0, self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Entity lists
        self.stars = [Star(self.WIDTH, self.HEIGHT) for _ in range(150)]
        self.player_projectiles = []
        self.alien_projectiles = []
        self.aliens = []
        self.particles = []
        
        # Alien spawning state
        self.aliens_to_spawn = self.TOTAL_ALIENS
        self.aliens_destroyed = 0
        self.alien_spawn_timer = 0
        self.alien_spawn_interval = 45 # frames
        
        # Difficulty state
        self.alien_speed = 1.5
        self.alien_fire_rate = 0.008 # P(fire) per alien per frame
        self.difficulty_tier = 0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.1  # Survival reward

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(action)
        self._update_game_state()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self.aliens_destroyed >= self.TOTAL_ALIENS:
                reward += 100  # Win bonus
                self.score += 5000
            if self.player_lives <= 0:
                reward -= 100  # Lose penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        if movement == 1:  # Up
            self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos.y += self.PLAYER_SPEED

        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)
        self.player_rect.center = self.player_pos

        if space_held and self.player_fire_cooldown == 0:
            # SFX: Player shoot
            self.player_projectiles.append(Projectile(
                self.player_pos + (self.PLAYER_RADIUS, 0), (12, 0), self.COLOR_PLAYER_PROJECTILE, 5
            ))
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN_MAX

    def _update_game_state(self):
        # Timers
        if self.player_fire_cooldown > 0: self.player_fire_cooldown -= 1
        if self.player_iframes > 0: self.player_iframes -= 1

        # Stars
        for star in self.stars: star.update(self.WIDTH)
        # Projectiles
        for p in self.player_projectiles[:]:
            p.update()
            if p.pos.x > self.WIDTH: self.player_projectiles.remove(p)
        for p in self.alien_projectiles[:]:
            p.update()
            if p.pos.x < 0: self.alien_projectiles.remove(p)
        # Particles
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0: self.particles.remove(p)
        # Aliens
        self._spawn_aliens()
        for alien in self.aliens[:]:
            alien.update()
            if alien.pos.x < -alien.radius:
                self.aliens.remove(alien)
            # Alien firing logic
            if self.np_random.random() < self.alien_fire_rate:
                # SFX: Alien shoot
                direction = (self.player_pos - alien.pos).normalize() if (self.player_pos - alien.pos).length() > 0 else pygame.Vector2(-1, 0)
                self.alien_projectiles.append(Projectile(
                    alien.pos, direction * 5, self.COLOR_ALIEN_PROJECTILE, 4
                ))

    def _spawn_aliens(self):
        self.alien_spawn_timer += 1
        if self.alien_spawn_timer >= self.alien_spawn_interval and self.aliens_to_spawn > 0:
            self.alien_spawn_timer = 0
            self.aliens_to_spawn -= 1
            self.aliens.append(Alien(self.alien_speed, self.alien_fire_rate))

    def _handle_collisions(self):
        reward = 0
        # Player projectiles vs Aliens
        for p in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if p.rect.colliderect(alien.rect):
                    # SFX: Alien hit
                    self._create_explosion(p.pos, 5, self.COLOR_EXPLOSION_ALIEN)
                    if p in self.player_projectiles: self.player_projectiles.remove(p)
                    
                    alien.health -= 1
                    reward += 1.0  # Hit reward
                    self.score += 10
                    if alien.health <= 0:
                        # SFX: Alien explosion
                        self._create_explosion(alien.pos, 20, self.COLOR_EXPLOSION_ALIEN)
                        self.aliens.remove(alien)
                        reward += 10.0  # Destroy reward
                        self.score += 100
                        self.aliens_destroyed += 1
                        self._update_difficulty()
                    break

        # Alien projectiles vs Player
        if self.player_iframes == 0:
            for p in self.alien_projectiles[:]:
                if p.rect.colliderect(self.player_rect):
                    # SFX: Player hit/explosion
                    self._create_explosion(self.player_pos, 30, self.COLOR_EXPLOSION_PLAYER)
                    self.alien_projectiles.remove(p)
                    self.player_lives -= 1
                    self.player_iframes = self.PLAYER_IFRAME_DURATION
                    reward -= 0.2  # Hit penalty
                    break
        return reward

    def _update_difficulty(self):
        new_tier = self.aliens_destroyed // 10
        if new_tier > self.difficulty_tier:
            self.difficulty_tier = new_tier
            self.alien_speed += 0.25
            self.alien_fire_rate += 0.001

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            self.particles.append(Particle(pos, color))

    def _check_termination(self):
        return self.player_lives <= 0 or self.aliens_destroyed >= self.TOTAL_ALIENS or self.steps >= self.MAX_STEPS

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
            "lives": self.player_lives,
            "aliens_destroyed": self.aliens_destroyed,
            "aliens_remaining": len(self.aliens) + self.aliens_to_spawn
        }

    def _render_game(self):
        for star in self.stars: star.draw(self.screen)
        for p in self.player_projectiles: p.draw(self.screen)
        for p in self.alien_projectiles: p.draw(self.screen)
        for alien in self.aliens: alien.draw(self.screen)
        for p in self.particles: p.draw(self.screen)
        self._render_player()

    def _render_player(self):
        if self.player_lives > 0:
            # Draw player only if not invincible or if invincible and frame is even
            if self.player_iframes == 0 or (self.player_iframes > 0 and self.steps % 4 < 2):
                px, py = int(self.player_pos.x), int(self.player_pos.y)
                
                # Glow effect
                glow_surface = self.screen.copy()
                glow_surface.set_colorkey((0,0,0))
                pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, (px, py), self.PLAYER_RADIUS + 5)
                glow_surface.set_alpha(100)
                self.screen.blit(glow_surface, (0,0))

                # Ship body
                points = [
                    (px + self.PLAYER_RADIUS, py),
                    (px - self.PLAYER_RADIUS, py - self.PLAYER_RADIUS * 0.8),
                    (px - self.PLAYER_RADIUS * 0.5, py),
                    (px - self.PLAYER_RADIUS, py + self.PLAYER_RADIUS * 0.8)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Aliens remaining
        aliens_text = self.font_small.render(f"ALIENS: {len(self.aliens) + self.aliens_to_spawn}", True, self.COLOR_UI)
        self.screen.blit(aliens_text, (self.WIDTH // 2 - aliens_text.get_width() // 2, 10))

        # Lives
        for i in range(self.player_lives):
            self._draw_heart(15 + i * 25, 18)
        
        # Game Over message
        if self.game_over:
            msg = "YOU WIN!" if self.aliens_destroyed >= self.TOTAL_ALIENS else "GAME OVER"
            color = self.COLOR_PLAYER if self.aliens_destroyed >= self.TOTAL_ALIENS else self.COLOR_HEART
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2))

    def _draw_heart(self, x, y):
        r = 8
        points = [
            (x, y + r//4), (x - r, y - r//2), (x - r//2, y - r), (x, y - r//2),
            (x + r//2, y - r), (x + r, y - r//2)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HEART)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HEART)

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Use a dummy window to display the game.
    pygame.display.set_caption("Galactic Defender")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Store the state of the keys
    keys_held = {
        "up": False,
        "down": False,
        "space": False,
        "shift": False
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: keys_held["up"] = True
                if event.key == pygame.K_DOWN: keys_held["down"] = True
                if event.key == pygame.K_SPACE: keys_held["space"] = True
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_held["shift"] = True
                if event.key == pygame.K_r: # Press R to reset
                    obs, info = env.reset()
                    done = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP: keys_held["up"] = False
                if event.key == pygame.K_DOWN: keys_held["down"] = False
                if event.key == pygame.K_SPACE: keys_held["space"] = False
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_held["shift"] = False

        # Construct the action from the key states
        movement_action = 0
        if keys_held["up"]: movement_action = 1
        elif keys_held["down"]: movement_action = 2
        
        space_action = 1 if keys_held["space"] else 0
        shift_action = 1 if keys_held["shift"] else 0
        
        action = [movement_action, space_action, shift_action]
        
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the observation to the screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Run at 30 FPS

    env.close()