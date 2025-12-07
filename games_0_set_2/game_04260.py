
# Generated: 2025-08-28T01:52:13.825089
# Source Brief: brief_04260.md
# Brief Index: 4260

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to fire your weapon."
    )

    # Short, user-facing description of the game
    game_description = (
        "Control a jumping, shooting robot in a side-scrolling arena to defeat waves of enemies."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_WIDTH = 1280
        self.PLATFORM_HEIGHT = 60
        self.FPS = 30
        self.MAX_STEPS = 10000

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_PLATFORM = (80, 80, 100)
        self.COLOR_PLAYER = (0, 160, 255)
        self.COLOR_ENEMY = (255, 65, 54)
        self.COLOR_PLAYER_PROJ = (255, 220, 0)
        self.COLOR_ENEMY_PROJ = (255, 133, 27)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_FULL = (46, 204, 64)
        self.COLOR_HEALTH_EMPTY = (133, 20, 75)
        
        # Physics
        self.GRAVITY = 0.8
        self.PLAYER_SPEED = 5
        self.PLAYER_JUMP_STRENGTH = -15
        self.PLAYER_FRICTION = 0.85

        # Gameplay
        self.PLAYER_MAX_HEALTH = 100
        self.ENEMY_MAX_HEALTH = 20
        self.PLAYER_SHOOT_COOLDOWN = 8 # frames
        self.ENEMY_SHOOT_COOLDOWN = 60 # frames (2 seconds)
        self.ENEMIES_PER_WAVE = 10
        self.ENEMY_PROJ_SPEED_BASE = 2.0
        self.ENEMY_PROJ_SPEED_WAVE_SCALING = 0.4
        self.PROJECTILE_SPEED = 12
        self.PROJECTILE_DAMAGE = 10

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 12)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_small = pygame.font.SysFont(None, 16)

        # --- Game State Initialization ---
        self.wave = 1
        self.last_run_won = False
        self.reset()
        
        # --- Self-Validation ---
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.last_run_won:
            self.wave += 1
        else:
            self.wave = 1
        self.last_run_won = False

        # State variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player
        self.player_pos = np.array([self.WORLD_WIDTH / 2, self.HEIGHT - self.PLATFORM_HEIGHT - 30], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_on_ground = False
        self.player_shoot_cooldown_timer = 0

        # Enemies
        self.enemies = []
        for _ in range(self.ENEMIES_PER_WAVE):
            self.enemies.append({
                "pos": np.array([
                    self.np_random.uniform(50, self.WORLD_WIDTH - 50),
                    self.HEIGHT - self.PLATFORM_HEIGHT - 20
                ], dtype=float),
                "health": self.ENEMY_MAX_HEALTH,
                "shoot_cooldown": self.np_random.integers(0, self.ENEMY_SHOOT_COOLDOWN)
            })

        # Projectiles & Particles
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        # Camera
        self.camera_x = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.001 # Small reward for surviving a frame

        # --- 1. Handle Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal movement
        if movement == 3: # Left
            self.player_vel[0] = -self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_vel[0] = self.PLAYER_SPEED
        
        # Jumping
        if movement == 1 and self.player_on_ground: # Up
            self.player_vel[1] = self.PLAYER_JUMP_STRENGTH
            self.player_on_ground = False
            # sfx: jump
            self._create_particles(self.player_pos + [0, 20], 5, self.COLOR_PLATFORM, count=5, speed_range=(1,3))


        # Shooting
        if space_held and self.player_shoot_cooldown_timer <= 0:
            self.player_shoot_cooldown_timer = self.PLAYER_SHOOT_COOLDOWN
            
            # Determine direction based on last movement or nearest enemy
            target_dir = np.array([1.0, 0.0]) # Default right
            if len(self.enemies) > 0:
                closest_enemy = min(self.enemies, key=lambda e: np.linalg.norm(e["pos"] - self.player_pos))
                if closest_enemy["pos"][0] < self.player_pos[0]:
                    target_dir = np.array([-1.0, 0.0])
            
            proj_start_pos = self.player_pos.copy() + target_dir * 20
            self.player_projectiles.append({"pos": proj_start_pos, "vel": target_dir * self.PROJECTILE_SPEED})
            # sfx: player_shoot
            self._create_particles(proj_start_pos, 3, self.COLOR_PLAYER_PROJ, count=8, speed_range=(2,5), lifespan=5)

        # --- 2. Update Game State ---
        self.steps += 1
        if self.player_shoot_cooldown_timer > 0:
            self.player_shoot_cooldown_timer -= 1
        
        # Player physics
        self.player_vel[0] *= self.PLAYER_FRICTION
        self.player_vel[1] += self.GRAVITY
        self.player_pos += self.player_vel
        self.player_on_ground = False
        
        # Collision: Player with platform
        if self.player_pos[1] >= self.HEIGHT - self.PLATFORM_HEIGHT - 20:
            self.player_pos[1] = self.HEIGHT - self.PLATFORM_HEIGHT - 20
            if self.player_vel[1] > 1: # Landing puff
                self._create_particles(self.player_pos + [0, 20], 5, self.COLOR_PLATFORM, count=5, speed_range=(1,3))
            self.player_vel[1] = 0
            self.player_on_ground = True
            
        # Collision: Player with world bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 10, self.WORLD_WIDTH - 10)

        # Enemy logic
        for enemy in self.enemies:
            # Movement
            direction_to_player = (self.player_pos[0] - enemy["pos"][0])
            if abs(direction_to_player) > 50: # Don't crowd player
                 enemy["pos"][0] += np.sign(direction_to_player) * 0.5
            
            # Shooting
            enemy["shoot_cooldown"] -= 1
            if enemy["shoot_cooldown"] <= 0:
                enemy["shoot_cooldown"] = self.ENEMY_SHOOT_COOLDOWN + self.np_random.integers(-15, 15)
                direction = self.player_pos - enemy["pos"]
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                speed = self.ENEMY_PROJ_SPEED_BASE + (self.wave - 1) * self.ENEMY_PROJ_SPEED_WAVE_SCALING
                self.enemy_projectiles.append({"pos": enemy["pos"].copy(), "vel": direction * speed})
                # sfx: enemy_shoot

        # Update player projectiles & check hits
        new_player_projectiles = []
        for proj in self.player_projectiles:
            proj["pos"] += proj["vel"]
            hit = False
            for enemy in self.enemies:
                if np.linalg.norm(proj["pos"] - enemy["pos"]) < 20: # Hitbox check
                    enemy["health"] -= self.PROJECTILE_DAMAGE
                    reward += 0.1
                    hit = True
                    self._create_particles(proj["pos"], 3, self.COLOR_ENEMY, count=10) # sfx: hit_confirm
                    break
            if not hit and 0 < proj["pos"][0] < self.WORLD_WIDTH and 0 < proj["pos"][1] < self.HEIGHT:
                new_player_projectiles.append(proj)
        self.player_projectiles = new_player_projectiles

        # Update enemy projectiles & check hits
        new_enemy_projectiles = []
        player_rect = pygame.Rect(self.player_pos[0]-10, self.player_pos[1]-20, 20, 40)
        for proj in self.enemy_projectiles:
            proj["pos"] += proj["vel"]
            hit = False
            if player_rect.collidepoint(proj["pos"]):
                self.player_health -= self.PROJECTILE_DAMAGE
                reward -= 0.2 # Small penalty for getting hit
                hit = True
                self._create_particles(proj["pos"], 3, self.COLOR_PLAYER, count=10) # sfx: player_hit
            if not hit and 0 < proj["pos"][0] < self.WORLD_WIDTH and 0 < proj["pos"][1] < self.HEIGHT:
                new_enemy_projectiles.append(proj)
        self.enemy_projectiles = new_enemy_projectiles
        
        # Process defeated enemies
        enemies_alive = []
        for enemy in self.enemies:
            if enemy["health"] > 0:
                enemies_alive.append(enemy)
            else:
                self.score += 100
                reward += 10
                self._create_particles(enemy["pos"], 8, self.COLOR_ENEMY, count=50, speed_range=(1,8), lifespan=40) # sfx: enemy_explode
        self.enemies = enemies_alive

        # Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] -= 0.1

        # --- 3. Check Termination ---
        terminated = False
        if self.player_health <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            self.last_run_won = False
            self._create_particles(self.player_pos, 8, self.COLOR_PLAYER, count=50, speed_range=(1,8), lifespan=40) # sfx: player_explode
        
        if not self.enemies:
            reward += 100
            terminated = True
            self.game_over = True
            self.last_run_won = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        # Update camera to follow player
        self.camera_x = self.player_pos[0] - self.WIDTH / 2
        self.camera_x = np.clip(self.camera_x, 0, self.WORLD_WIDTH - self.WIDTH)

        # --- Render everything ---
        self._render_background()
        self._render_game_objects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Draw grid
        grid_offset = -self.camera_x % 40
        for x in range(int(grid_offset) - 40, self.WIDTH + 40, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        # Draw platform
        platform_rect = pygame.Rect(0, self.HEIGHT - self.PLATFORM_HEIGHT, self.WIDTH, self.PLATFORM_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, platform_rect)
        pygame.draw.rect(self.screen, tuple(c*1.2 for c in self.COLOR_PLATFORM), platform_rect, 3)

    def _render_game_objects(self):
        # Particles
        for p in self.particles:
            if p['size'] > 0:
                pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), p['color'] + (int(p['lifespan']/p['max_lifespan']*255),))
        
        # Projectiles
        for proj in self.player_projectiles:
            pos = (int(proj['pos'][0] - self.camera_x), int(proj['pos'][1]))
            end_pos = (int((proj['pos']-proj['vel'])[0] - self.camera_x), int((proj['pos']-proj['vel'])[1]))
            pygame.draw.line(self.screen, self.COLOR_PLAYER_PROJ, pos, end_pos, 4)
        for proj in self.enemy_projectiles:
            pos = (int(proj['pos'][0] - self.camera_x), int(proj['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_ENEMY_PROJ)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, self.COLOR_ENEMY_PROJ)

        # Enemies
        for enemy in self.enemies:
            x, y = enemy['pos'] - np.array([self.camera_x, 0])
            body = pygame.Rect(x-10, y-15, 20, 30)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, body, border_radius=3)
            # Health bar
            health_pct = enemy['health'] / self.ENEMY_MAX_HEALTH
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_EMPTY, (x-12, y-25, 24, 4))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FULL, (x-12, y-25, 24 * health_pct, 4))

        # Player
        if self.player_health > 0:
            x, y = self.player_pos - np.array([self.camera_x, 0])
            # Squash and stretch based on vertical velocity
            squash = 1 - min(max(self.player_vel[1] / 30, -0.4), 0.4)
            stretch = 1 + min(max(self.player_vel[1] / 30, -0.4), 0.4)
            w, h = 20 * squash, 40 * stretch
            
            body = pygame.Rect(x - w/2, y - h/2, w, h)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, body, border_radius=4)
            # "Eye"
            eye_dir = 1 if self.player_vel[0] >= 0 else -1
            if len(self.enemies) > 0:
                closest_enemy = min(self.enemies, key=lambda e: np.linalg.norm(e["pos"] - self.player_pos))
                if closest_enemy["pos"][0] < self.player_pos[0]: eye_dir = -1
                else: eye_dir = 1
            pygame.draw.circle(self.screen, self.COLOR_BG, (int(x + 5 * eye_dir), int(y-5)), 3)
            
            # Health bar
            health_pct = self.player_health / self.PLAYER_MAX_HEALTH
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_EMPTY, (x-20, y-35, 40, 5))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FULL, (x-20, y-35, 40 * health_pct, 5))

    def _render_ui(self):
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        wave_surf = self.font_ui.render(f"WAVE: {self.wave}", True, self.COLOR_TEXT)
        self.screen.blit(wave_surf, (self.WIDTH/2 - wave_surf.get_width()/2, 10))
        
        enemies_surf = self.font_ui.render(f"ENEMIES: {len(self.enemies)}", True, self.COLOR_TEXT)
        self.screen.blit(enemies_surf, (self.WIDTH - enemies_surf.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player_health,
            "enemies_left": len(self.enemies)
        }

    def _create_particles(self, pos, size, color, count=20, speed_range=(1, 5), lifespan=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(*speed_range)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy() + self.np_random.uniform(-3, 3, size=2),
                'vel': vel,
                'size': size * self.np_random.uniform(0.5, 1.5),
                'color': color,
                'lifespan': lifespan,
                'max_lifespan': lifespan
            })

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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play ---
    # This part allows a human to play the game.
    # It will only run if the script is executed directly.
    
    # Set render mode to 'human' if available, or just display the array
    try:
        pygame.display.set_caption("GameEnv Manual Test")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        human_render = True
    except pygame.error:
        print("No display available. Manual play is disabled.")
        human_render = False

    if human_render:
        obs, info = env.reset()
        done = False
        
        while not done:
            # Action defaults
            movement, space, shift = 0, 0, 0
            
            # Pygame event loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            # Key polling for continuous actions
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            if keys[pygame.K_DOWN]:
                movement = 2 # In this game, down does nothing
            if keys[pygame.K_LEFT]:
                movement = 3
            if keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift = 1

            action = [movement, space, shift]
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")

            # Render to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Cap the frame rate
            env.clock.tick(env.FPS)
            
    env.close()