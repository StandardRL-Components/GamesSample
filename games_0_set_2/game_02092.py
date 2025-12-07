
# Generated: 2025-08-27T19:14:35.085562
# Source Brief: brief_02092.md
# Brief Index: 2092

        
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


# Helper classes for game objects
class Robot:
    def __init__(self, pos):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.health = 100
        self.max_health = 100
        self.width = 40
        self.height = 60
        self.on_ground = True
        self.facing_left = False
        self.is_attacking = False
        self.attack_timer = 0
        self.attack_cooldown = 0
        self.damage_flash_timer = 0
        self.rect = pygame.Rect(self.pos.x, self.pos.y, self.width, self.height)

    def update_rect(self):
        self.rect.topleft = (self.pos.x, self.pos.y)

class Enemy:
    def __init__(self, pos, speed):
        self.pos = pygame.Vector2(pos)
        self.speed = speed
        self.health = 20
        self.max_health = 20
        self.width = 30
        self.height = 40
        self.damage_flash_timer = 0
        self.rect = pygame.Rect(self.pos.x, self.pos.y, self.width, self.height)

    def update_rect(self):
        self.rect.topleft = (self.pos.x, self.pos.y)

class Particle:
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.radius = max(0, self.radius - 0.2)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to attack."
    )

    game_description = (
        "Control a powerful robot in a side-scrolling arcade environment, crushing waves of enemies to achieve total domination."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GROUND_Y = self.SCREEN_HEIGHT - 60

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_wave = pygame.font.SysFont("monospace", 32, bold=True)

        # Colors
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_GROUND = (40, 30, 50)
        self.COLOR_CITY_1 = (25, 20, 40)
        self.COLOR_CITY_2 = (35, 30, 45)
        self.COLOR_ROBOT = (60, 120, 255)
        self.COLOR_ENEMY = (255, 80, 60)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_HEALTH_BG = (100, 20, 20)
        self.COLOR_HEALTH_FG = (20, 200, 20)
        
        # Physics and Game constants
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = 15
        self.ROBOT_SPEED = 5
        self.ENEMY_BASE_SPEED = 1.0
        self.ROBOT_ATTACK_DAMAGE = 20
        self.ENEMY_CONTACT_DAMAGE = 5
        self.WAVE_SIZE = 12
        self.MAX_STEPS = 1000

        # State variables are initialized in reset()
        self.robot = None
        self.enemies = []
        self.particles = []
        self.cityscape = []
        self.steps = 0
        self.score = 0
        self.wave_number = 1
        self.game_over = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.wave_number = 1
        self.game_over = False

        self.robot = Robot(pos=(self.SCREEN_WIDTH / 2, self.GROUND_Y - 60))
        self.enemies = []
        self.particles = []

        self._spawn_wave()
        self._generate_cityscape()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0
        terminated = False
        
        if not self.game_over:
            # 1. Unpack action and handle player input
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            reward += self._handle_player_actions(movement, space_held)

            # 2. Update physics and timers
            self._update_player_physics()
            self._update_timers()

            # 3. Update enemies
            self._update_enemies()

            # 4. Handle collisions and damage
            reward += self._handle_collisions()

            # 5. Update particles and cleanup dead enemies
            self._update_particles()
            reward += self._cleanup_enemies()
            
            # 6. Check for wave clear
            if not self.enemies:
                self.wave_number += 1
                self._spawn_wave()
                reward += 100 # Wave clear bonus
                # Per brief, episode terminates on wave clear
                self.game_over = True
                terminated = True

            # 7. Check termination conditions
            self.steps += 1
            if self.robot.health <= 0:
                reward -= 100 # Death penalty
                self.game_over = True
                terminated = True
            elif self.steps >= self.MAX_STEPS:
                terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_actions(self, movement, space_held):
        # Calculate distance to nearest enemy for reward
        dist_before = self._get_dist_to_nearest_enemy()

        # Horizontal Movement
        self.robot.vel.x = 0
        if movement == 3:  # Left
            self.robot.vel.x = -self.ROBOT_SPEED
            self.robot.facing_left = True
        elif movement == 4:  # Right
            self.robot.vel.x = self.ROBOT_SPEED
            self.robot.facing_left = False

        # Jumping
        if movement == 1 and self.robot.on_ground:  # Up
            self.robot.vel.y = -self.JUMP_STRENGTH
            self.robot.on_ground = False
            # sfx: jump_sound

        # Attacking
        if space_held and self.robot.attack_cooldown == 0:
            self.robot.is_attacking = True
            self.robot.attack_timer = 5  # Attack lasts 5 frames
            self.robot.attack_cooldown = 15  # Cooldown of 15 frames
            # sfx: attack_whoosh

        # Positional reward
        dist_after = self._get_dist_to_nearest_enemy(after_move=True)
        if dist_after < dist_before:
            return 0.1
        elif dist_after > dist_before:
            return -0.1
        return 0

    def _update_player_physics(self):
        # Apply gravity
        if not self.robot.on_ground:
            self.robot.vel.y += self.GRAVITY
        
        # Update position
        self.robot.pos += self.robot.vel

        # Ground collision
        if self.robot.pos.y + self.robot.height > self.GROUND_Y:
            self.robot.pos.y = self.GROUND_Y - self.robot.height
            self.robot.vel.y = 0
            self.robot.on_ground = True

        # Screen bounds
        self.robot.pos.x = max(0, min(self.robot.pos.x, self.SCREEN_WIDTH - self.robot.width))
        self.robot.update_rect()

    def _update_timers(self):
        self.robot.attack_cooldown = max(0, self.robot.attack_cooldown - 1)
        self.robot.attack_timer = max(0, self.robot.attack_timer - 1)
        self.robot.damage_flash_timer = max(0, self.robot.damage_flash_timer - 1)
        if self.robot.attack_timer == 0:
            self.robot.is_attacking = False
        for enemy in self.enemies:
            enemy.damage_flash_timer = max(0, enemy.damage_flash_timer - 1)

    def _update_enemies(self):
        for enemy in self.enemies:
            direction = 1 if self.robot.pos.x > enemy.pos.x else -1
            enemy.pos.x += direction * enemy.speed
            enemy.pos.x = max(0, min(enemy.pos.x, self.SCREEN_WIDTH - enemy.width))
            enemy.update_rect()

    def _handle_collisions(self):
        reward = 0
        attack_rect = self._get_attack_rect()

        for enemy in self.enemies:
            # Robot attack hits enemy
            if self.robot.is_attacking and attack_rect and enemy.rect.colliderect(attack_rect):
                if not hasattr(enemy, 'hit_this_attack'): # Prevent multiple hits per swing
                    enemy.health -= self.ROBOT_ATTACK_DAMAGE
                    enemy.damage_flash_timer = 5
                    enemy.hit_this_attack = True
                    reward += 1 # Damage reward
                    # sfx: enemy_hit_sound

            # Enemy hits robot
            if self.robot.rect.colliderect(enemy.rect) and self.robot.damage_flash_timer == 0:
                self.robot.health -= self.ENEMY_CONTACT_DAMAGE
                self.robot.health = max(0, self.robot.health)
                self.robot.damage_flash_timer = 20 # Invincibility frames
                reward -= 1 # Damage taken penalty
                # sfx: robot_damage_sound
        
        # Reset hit flags after checking all enemies for this attack frame
        if not self.robot.is_attacking:
            for enemy in self.enemies:
                if hasattr(enemy, 'hit_this_attack'):
                    del enemy.hit_this_attack
        return reward

    def _cleanup_enemies(self):
        destroyed_enemies = [e for e in self.enemies if e.health <= 0]
        self.enemies = [e for e in self.enemies if e.health > 0]
        
        for enemy in destroyed_enemies:
            self.score += 10
            self._create_explosion(enemy.rect.center)
            # sfx: explosion_sound
        return 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background
        self._render_cityscape()
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

        # Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), int(p.radius), p.color)

        # Enemies
        for enemy in self.enemies:
            color = self.COLOR_WHITE if enemy.damage_flash_timer > 0 else self.COLOR_ENEMY
            pygame.draw.rect(self.screen, color, enemy.rect)
            self._render_health_bar(enemy.pos, enemy.health, enemy.max_health, enemy.width)

        # Robot
        robot_color = self.COLOR_WHITE if self.robot.damage_flash_timer > 0 else self.COLOR_ROBOT
        pygame.draw.rect(self.screen, robot_color, self.robot.rect)
        if self.robot.is_attacking:
            attack_rect = self._get_attack_rect()
            if attack_rect:
                pygame.draw.rect(self.screen, self.COLOR_WHITE, attack_rect)

    def _render_ui(self):
        # Robot Health
        health_ratio = self.robot.health / self.robot.max_health
        health_bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (10, 10, health_bar_width * health_ratio, 20))
        health_text = self.font_ui.render(f"HP", True, self.COLOR_WHITE)
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Wave
        wave_text = self.font_wave.render(f"WAVE {self.wave_number}", True, self.COLOR_WHITE)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH / 2 - wave_text.get_width() / 2, 10))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number}

    def _spawn_wave(self):
        enemy_speed = self.ENEMY_BASE_SPEED + (self.wave_number - 1) * 0.1
        for _ in range(self.WAVE_SIZE):
            side = self.np_random.choice([-1, 1])
            x_pos = -50 if side == -1 else self.SCREEN_WIDTH + 50
            y_pos = self.GROUND_Y - 40
            self.enemies.append(Enemy(pos=(x_pos, y_pos), speed=enemy_speed))

    def _generate_cityscape(self):
        self.cityscape = []
        for _ in range(20): # Far layer
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            h = self.np_random.integers(50, 150)
            w = self.np_random.integers(20, 60)
            self.cityscape.append({'rect': pygame.Rect(x, self.GROUND_Y - h, w, h), 'color': self.COLOR_CITY_1})
        for _ in range(15): # Near layer
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            h = self.np_random.integers(80, 200)
            w = self.np_random.integers(30, 80)
            self.cityscape.append({'rect': pygame.Rect(x, self.GROUND_Y - h, w, h), 'color': self.COLOR_CITY_2})

    def _render_cityscape(self):
        for building in self.cityscape:
            pygame.draw.rect(self.screen, building['color'], building['rect'])
            
    def _render_health_bar(self, pos, current, maximum, width):
        if current < maximum:
            ratio = current / maximum
            bar_y = pos.y - 10
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (pos.x, bar_y, width, 5))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (pos.x, bar_y, width * ratio, 5))

    def _get_attack_rect(self):
        if not self.robot.is_attacking:
            return None
        attack_width, attack_height = 50, 10
        if self.robot.facing_left:
            return pygame.Rect(self.robot.rect.left - attack_width, self.robot.rect.centery - attack_height/2, attack_width, attack_height)
        else:
            return pygame.Rect(self.robot.rect.right, self.robot.rect.centery - attack_height/2, attack_width, attack_height)

    def _create_explosion(self, position):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            radius = self.np_random.uniform(3, 8)
            color = random.choice([(255, 255, 100), (255, 180, 50), (255, 255, 255)])
            lifespan = self.np_random.integers(15, 30)
            self.particles.append(Particle(position, vel, radius, color, lifespan))

    def _update_particles(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifespan > 0]

    def _get_dist_to_nearest_enemy(self, after_move=False):
        if not self.enemies:
            return 0
        
        robot_pos = self.robot.pos + self.robot.vel if after_move else self.robot.pos
        
        min_dist_sq = float('inf')
        for enemy in self.enemies:
            dist_sq = (robot_pos - enemy.pos).length_squared()
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        return math.sqrt(min_dist_sq)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Robot Brawler")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    
    # Mapping from Pygame keys to your environment's actions
    key_to_action = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
    }

    while running:
        # --- Human Input ---
        movement_action = 0  # Default to no-op for movement
        space_action = 0     # Default to released
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        elif keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2

        if keys[pygame.K_SPACE]:
            space_action = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                terminated = False

        if terminated:
            # Show a "Game Over" or "Wave Cleared" message
            font = pygame.font.SysFont("monospace", 50, bold=True)
            text = font.render("GAME OVER", True, (255, 0, 0))
            if not env.robot.health <= 0: # If not dead, must be wave clear
                text = font.render("WAVE CLEARED!", True, (0, 255, 0))
            
            text_rect = text.get_rect(center=(env.SCREEN_WIDTH/2, env.SCREEN_HEIGHT/2))
            
            # Draw the last frame
            frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(frame_surface, (0, 0))
            
            # Overlay the text
            screen.blit(text, text_rect)
            
            # Prompt to reset
            font_small = pygame.font.SysFont("monospace", 20, bold=True)
            reset_text = font_small.render("Press 'R' to restart", True, (255, 255, 255))
            reset_rect = reset_text.get_rect(center=(env.SCREEN_WIDTH/2, env.SCREEN_HEIGHT/2 + 50))
            screen.blit(reset_text, reset_rect)

            pygame.display.flip()
            continue

        # --- Environment Step ---
        action = [movement_action, space_action, 0] # Shift is not used
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # Convert the observation (numpy array) back to a Pygame surface
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

    env.close()