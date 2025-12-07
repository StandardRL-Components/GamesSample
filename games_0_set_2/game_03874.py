
# Generated: 2025-08-28T00:42:06.456497
# Source Brief: brief_03874.md
# Brief Index: 3874

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game entities
class Player:
    def __init__(self, pos, size, color):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.size = size
        self.color = color
        self.health = 100
        self.max_health = 100
        self.facing_direction = pygame.Vector2(1, 0) # Start facing right

class Enemy:
    def __init__(self, pos, size, color, pattern_center, pattern_radius, pattern_speed):
        self.pos = pygame.Vector2(pos)
        self.size = size
        self.color = color
        self.health = 20
        self.max_health = 20
        self.fire_cooldown = 0
        
        # Movement pattern
        self.pattern_center = pygame.Vector2(pattern_center)
        self.pattern_radius = pattern_radius
        self.pattern_speed = pattern_speed
        self.pattern_angle = random.uniform(0, 2 * math.pi)
        self.bob_angle = random.uniform(0, 2 * math.pi)
        self.bob_speed = 3.0
        self.bob_amount = 5

    def update(self, dt):
        # Circular pattern
        self.pattern_angle += self.pattern_speed * dt
        offset_x = math.cos(self.pattern_angle) * self.pattern_radius
        offset_y = math.sin(self.pattern_angle) * self.pattern_radius
        
        # Bobbing motion
        self.bob_angle += self.bob_speed * dt
        bob_offset = math.sin(self.bob_angle) * self.bob_amount
        
        self.pos.x = self.pattern_center.x + offset_x
        self.pos.y = self.pattern_center.y + offset_y + bob_offset

class Projectile:
    def __init__(self, pos, vel, color, size, lifetime):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.size = size
        self.lifetime = lifetime

class Particle:
    def __init__(self, pos, color, start_radius, end_radius, lifetime):
        self.pos = pygame.Vector2(pos)
        self.color = color
        self.start_radius = start_radius
        self.end_radius = end_radius
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.lifetime -= 1
        return self.lifetime > 0

    def get_current_radius(self):
        progress = 1.0 - (self.lifetime / self.max_lifetime)
        return self.start_radius + (self.end_radius - self.start_radius) * progress

    def get_current_alpha(self):
        return max(0, 255 * (self.lifetime / self.max_lifetime))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Hold space to fire. Survive the onslaught!"
    )

    game_description = (
        "Pilot a mech in an isometric arena, blasting waves of enemies to achieve ultimate victory."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and World Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WORLD_WIDTH = 1000
        self.WORLD_HEIGHT = 1000

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        # Colors
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_PLAYER_PROJ = (255, 255, 0)
        self.COLOR_ENEMY_PROJ = (255, 100, 0)
        self.COLOR_EXPLOSION = (255, 150, 0)
        self.COLOR_HEALTH = (0, 200, 80)
        self.COLOR_UI_TEXT = (220, 220, 240)
        
        # Game constants
        self.MAX_STEPS = 5000
        self.PLAYER_SPEED = 8
        self.PLAYER_FIRE_COOLDOWN = 4
        self.PLAYER_PROJ_SPEED = 15
        self.ENEMY_PROJ_SPEED = 6
        self.TOTAL_ENEMY_COUNT = 25
        self.WAVE_SIZE = 5

        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.player = Player(
            pos=(self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2),
            size=20,
            color=self.COLOR_PLAYER
        )
        self.player_fire_timer = 0
        
        # Entity lists
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        # Wave and difficulty state
        self.total_enemies_spawned = 0
        self.enemy_base_fire_cooldown = 30
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.clock.tick(30) # Maintain 30 FPS for smooth visuals
        dt = 1.0 # Assume fixed delta time for simplicity
        reward = -0.01 # Small penalty for each step to encourage speed

        # --- ACTION HANDLING ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_player_input(movement, space_held, dt)
        
        # --- UPDATE GAME STATE ---
        self.player.pos += self.player.vel * dt
        self._clamp_to_world(self.player.pos)

        # Update enemies
        for enemy in self.enemies:
            enemy.update(dt)
            enemy.fire_cooldown -= 1
            if enemy.fire_cooldown <= 0:
                self._enemy_fire(enemy)
                enemy.fire_cooldown = self.enemy_base_fire_cooldown - int(self.steps / 500)

        # Update projectiles
        self._update_projectiles(self.player_projectiles, dt)
        self._update_projectiles(self.enemy_projectiles, dt)
        
        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        # --- COLLISION DETECTION ---
        reward += self._handle_collisions()

        # --- CHECK GAME FLOW ---
        if len(self.enemies) == 0 and self.total_enemies_spawned < self.TOTAL_ENEMY_COUNT:
            self._spawn_wave()

        self.steps += 1
        terminated = self._check_termination()

        # Add terminal rewards
        if terminated:
            if self.player.health <= 0:
                reward -= 100
            elif len(self.enemies) == 0 and self.total_enemies_spawned == self.TOTAL_ENEMY_COUNT:
                reward += 100
                self.score += 1000 # Victory bonus

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held, dt):
        self.player.vel = pygame.Vector2(0, 0)
        move_dir = pygame.Vector2(0, 0)
        if movement == 1: move_dir.y = -1 # Up
        elif movement == 2: move_dir.y = 1 # Down
        elif movement == 3: move_dir.x = -1 # Left
        elif movement == 4: move_dir.x = 1 # Right
        
        if move_dir.length() > 0:
            self.player.vel = move_dir.normalize() * self.PLAYER_SPEED
            self.player.facing_direction = move_dir.normalize()

        self.player_fire_timer -= 1
        if space_held and self.player_fire_timer <= 0:
            # sfx: Player shoot
            proj_pos = self.player.pos + self.player.facing_direction * (self.player.size)
            proj_vel = self.player.facing_direction * self.PLAYER_PROJ_SPEED
            self.player_projectiles.append(Projectile(proj_pos, proj_vel, self.COLOR_PLAYER_PROJ, 3, 60))
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

    def _enemy_fire(self, enemy):
        # sfx: Enemy shoot
        direction_to_player = (self.player.pos - enemy.pos).normalize()
        proj_pos = enemy.pos + direction_to_player * (enemy.size)
        proj_vel = direction_to_player * self.ENEMY_PROJ_SPEED
        self.enemy_projectiles.append(Projectile(proj_pos, proj_vel, self.COLOR_ENEMY_PROJ, 2, 100))

    def _update_projectiles(self, projectiles, dt):
        for proj in projectiles[:]:
            proj.pos += proj.vel * dt
            proj.lifetime -= 1
            if proj.lifetime <= 0 or not (0 < proj.pos.x < self.WORLD_WIDTH and 0 < proj.pos.y < self.WORLD_HEIGHT):
                projectiles.remove(proj)

    def _handle_collisions(self):
        reward = 0
        # Player projectiles vs Enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if proj.pos.distance_to(enemy.pos) < enemy.size:
                    # sfx: Hit confirm
                    enemy.health -= 20
                    reward += 0.1
                    self.particles.append(Particle(proj.pos, self.COLOR_EXPLOSION, 2, 10, 10))
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    
                    if enemy.health <= 0:
                        # sfx: Enemy explosion
                        self.particles.append(Particle(enemy.pos, self.COLOR_EXPLOSION, 5, 40, 20))
                        self.enemies.remove(enemy)
                        self.score += 100
                        reward += 1.0
                    break

        # Enemy projectiles vs Player
        for proj in self.enemy_projectiles[:]:
            if proj.pos.distance_to(self.player.pos) < self.player.size:
                # sfx: Player hit
                self.player.health -= 10
                self.particles.append(Particle(proj.pos, self.COLOR_PLAYER, 2, 15, 12))
                if proj in self.enemy_projectiles: self.enemy_projectiles.remove(proj)
                break
        
        self.player.health = max(0, self.player.health)
        return reward

    def _spawn_wave(self):
        num_to_spawn = min(self.WAVE_SIZE, self.TOTAL_ENEMY_COUNT - self.total_enemies_spawned)
        for _ in range(num_to_spawn):
            # Spawn in a ring around the center
            angle = self.np_random.uniform(0, 2 * math.pi)
            dist = self.np_random.uniform(300, 450)
            spawn_pos = (
                self.WORLD_WIDTH / 2 + math.cos(angle) * dist,
                self.WORLD_HEIGHT / 2 + math.sin(angle) * dist
            )
            
            # Define circular movement pattern
            pattern_center = pygame.Vector2(self.np_random.uniform(100, 900), self.np_random.uniform(100, 900))
            pattern_radius = self.np_random.uniform(50, 150)
            pattern_speed = self.np_random.uniform(0.02, 0.05)

            self.enemies.append(Enemy(
                pos=spawn_pos, 
                size=15, 
                color=self.COLOR_ENEMY,
                pattern_center=pattern_center,
                pattern_radius=pattern_radius,
                pattern_speed=pattern_speed
            ))
        self.total_enemies_spawned += num_to_spawn
        
    def _check_termination(self):
        if self.player.health <= 0:
            return True
        if len(self.enemies) == 0 and self.total_enemies_spawned >= self.TOTAL_ENEMY_COUNT:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player.health,
            "enemies_left": len(self.enemies) + (self.TOTAL_ENEMY_COUNT - self.total_enemies_spawned)
        }

    def _clamp_to_world(self, pos):
        pos.x = max(0, min(self.WORLD_WIDTH, pos.x))
        pos.y = max(0, min(self.WORLD_HEIGHT, pos.y))

    def _iso_to_screen(self, x, y):
        screen_x = self.SCREEN_WIDTH / 2 + (x - y) * 0.5
        screen_y = self.SCREEN_HEIGHT / 4 + (x + y) * 0.25
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw isometric grid
        for i in range(0, self.WORLD_WIDTH + 1, 100):
            start = self._iso_to_screen(i, 0)
            end = self._iso_to_screen(i, self.WORLD_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for i in range(0, self.WORLD_HEIGHT + 1, 100):
            start = self._iso_to_screen(0, i)
            end = self._iso_to_screen(self.WORLD_WIDTH, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        
        # Collect all renderable entities
        render_list = [self.player] + self.enemies + self.player_projectiles + self.enemy_projectiles
        render_list.sort(key=lambda e: e.pos.y) # Sort for proper occlusion
        
        for entity in render_list:
            sx, sy = self._iso_to_screen(entity.pos.x, entity.pos.y)
            if isinstance(entity, Player):
                pygame.draw.rect(self.screen, entity.color, (sx - entity.size//2, sy - entity.size//2, entity.size, entity.size))
            elif isinstance(entity, Enemy):
                points = [
                    self._iso_to_screen(entity.pos.x, entity.pos.y - entity.size),
                    self._iso_to_screen(entity.pos.x - entity.size, entity.pos.y + entity.size),
                    self._iso_to_screen(entity.pos.x + entity.size, entity.pos.y + entity.size),
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, entity.color)
                pygame.gfxdraw.aapolygon(self.screen, points, entity.color)
            elif isinstance(entity, Projectile):
                start_pos = self._iso_to_screen(entity.pos.x - entity.vel.x * 0.5, entity.pos.y - entity.vel.y * 0.5)
                end_pos = self._iso_to_screen(entity.pos.x, entity.pos.y)
                pygame.draw.line(self.screen, entity.color, start_pos, end_pos, entity.size)

        # Render particles on top
        for p in self.particles:
            sx, sy = self._iso_to_screen(p.pos.x, p.pos.y)
            radius = p.get_current_radius()
            alpha = p.get_current_alpha()
            if radius > 0:
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*p.color, alpha), (radius, radius), radius)
                self.screen.blit(temp_surf, (sx - radius, sy - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Health Bar
        health_pct = self.player.health / self.player.max_health
        bar_width = 200
        pygame.draw.rect(self.screen, (50,0,0), (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, bar_width * health_pct, 20))
        health_text = self.font_small.render(f"HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Enemy Counter
        enemies_left = len(self.enemies) + (self.TOTAL_ENEMY_COUNT - self.total_enemies_spawned)
        enemy_text = self.font_small.render(f"ENEMIES: {enemies_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(enemy_text, (self.SCREEN_WIDTH - enemy_text.get_width() - 10, 10))

        # Score
        score_text = self.font_large.render(f"{self.score:07d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH/2 - score_text.get_width()/2, self.SCREEN_HEIGHT - 40))

        # Game Over / Victory Text
        if self._check_termination():
            if self.player.health <= 0:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY)
            else:
                end_text = self.font_large.render("VICTORY", True, self.COLOR_PLAYER)
            self.screen.blit(end_text, (self.SCREEN_WIDTH/2 - end_text.get_width()/2, self.SCREEN_HEIGHT/2 - end_text.get_height()/2))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example usage:
if __name__ == '__main__':
    # To play the game manually
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use 'x11', 'dummy', 'windows', 'quartz' as appropriate
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Mech Arena")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Mapping keyboard keys to MultiDiscrete action components
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while not terminated:
        # --- Human Input ---
        movement_action = 0 # No-op
        space_action = 0 # Released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        for key, move_val in key_to_action.items():
            if keys[key]:
                movement_action = move_val
                break # Prioritize first key in dict order
        
        if keys[pygame.K_SPACE]:
            space_action = 1 # Held

        if keys[pygame.K_r]: # Press R to reset
             obs, info = env.reset()

        action = [movement_action, space_action, 0] # Last component (shift) is unused
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)
        
    env.close()