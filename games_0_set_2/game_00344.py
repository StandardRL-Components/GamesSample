
# Generated: 2025-08-27T13:22:23.451780
# Source Brief: brief_00344.md
# Brief Index: 344

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to fire your weapon."
    )

    game_description = (
        "Control a jumping, shooting robot to blast through enemy waves and reach the exit in this side-scrolling action platformer."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FLOOR_Y = self.HEIGHT - 50
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_TILE = (40, 50, 70)
        self.COLOR_PLAYER = (60, 255, 150)
        self.COLOR_ENEMY = (255, 80, 80)
        self.COLOR_PLAYER_PROJ = (255, 255, 100)
        self.COLOR_ENEMY_PROJ = (255, 100, 255)
        self.COLOR_EXIT = (100, 255, 255)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_HEALTH = (60, 255, 60)
        self.COLOR_HEALTH_BG = (100, 20, 20)
        
        # Physics & Game Rules
        self.GRAVITY = 0.5
        self.PLAYER_SPEED = 5.0
        self.JUMP_STRENGTH = -11.0
        self.MAX_HEALTH = 30
        self.MAX_STAGES = 3
        self.MAX_STEPS = 3000
        self.LEVEL_WIDTH_FACTOR = 4 # Level width is 4 * screen width
        self.LEVEL_WIDTH = self.WIDTH * self.LEVEL_WIDTH_FACTOR

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # --- State Variables ---
        self.seed = None
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        
        # Player
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_health = 0
        self.on_ground = False
        self.player_dir = 1
        
        # Camera
        self.camera_x = 0.0
        
        # Entities
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.exit_pos = np.array([0.0, 0.0])
        
        # Input handling
        self.last_space_held = False
        self.shoot_cooldown = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            # Fallback if no seed is provided
            if self.np_random is None:
                self.np_random, _ = gym.utils.seeding.np_random(0)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        
        self.player_pos = np.array([100.0, self.FLOOR_Y - 30.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_health = self.MAX_HEALTH
        self.on_ground = True
        self.player_dir = 1
        
        self.camera_x = 0.0
        self.last_space_held = False
        self.shoot_cooldown = 0

        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        self.exit_pos = np.array([self.LEVEL_WIDTH - 100.0, self.FLOOR_Y - 40.0])
        
        enemy_fire_rate = 90 - (self.stage - 1) * 3 # 3s, 2.9s, 2.8s at 30fps
        
        for i in range(10):
            spawn_x = self.WIDTH + i * (self.LEVEL_WIDTH - self.WIDTH) / 9.0
            patrol_center = spawn_x + self.np_random.uniform(-100, 100)
            self.enemies.append({
                "pos": np.array([patrol_center, self.FLOOR_Y - 25.0]),
                "health": 10,
                "patrol_min": patrol_center - 50,
                "patrol_max": patrol_center + 50,
                "dir": self.np_random.choice([-1, 1]),
                "shoot_timer": self.np_random.integers(0, enemy_fire_rate),
                "fire_rate": enemy_fire_rate
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.01  # Survival reward

        # --- Handle Input ---
        # Horizontal Movement
        if movement == 3: # Left
            self.player_vel[0] = -self.PLAYER_SPEED
            self.player_dir = -1
        elif movement == 4: # Right
            self.player_vel[0] = self.PLAYER_SPEED
            self.player_dir = 1
        else:
            self.player_vel[0] = 0

        # Jumping
        if movement == 1 and self.on_ground:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.on_ground = False
            # Sound: Jump
            self._create_particles(self.player_pos + np.array([15, 30]), 5, self.COLOR_WHITE, 0.5)

        # Shooting
        if space_held and not self.last_space_held and self.shoot_cooldown <= 0:
            proj_start_pos = self.player_pos + np.array([15 + self.player_dir * 20, 15])
            self.player_projectiles.append({
                "pos": proj_start_pos,
                "dir": self.player_dir
            })
            self.shoot_cooldown = 10 # 1/3 second cooldown at 30fps
            # Sound: Player Shoot
            self._create_particles(proj_start_pos, 10, self.COLOR_PLAYER_PROJ, 0.3, speed=5, direction=self.player_dir)
        self.last_space_held = space_held
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        # --- Update Player ---
        self.player_vel[1] += self.GRAVITY
        self.player_pos += self.player_vel
        
        # Floor collision
        if self.player_pos[1] > self.FLOOR_Y - 30:
            self.player_pos[1] = self.FLOOR_Y - 30
            self.player_vel[1] = 0
            self.on_ground = True
        
        # World bounds
        self.player_pos[0] = max(0, self.player_pos[0])

        # --- Update Enemies ---
        for enemy in self.enemies:
            enemy["pos"][0] += enemy["dir"] * 1.5
            if enemy["pos"][0] < enemy["patrol_min"]:
                enemy["dir"] = 1
            elif enemy["pos"][0] > enemy["patrol_max"]:
                enemy["dir"] = -1
            
            enemy["shoot_timer"] -= 1
            if enemy["shoot_timer"] <= 0:
                enemy_dir = -1 if self.player_pos[0] < enemy["pos"][0] else 1
                self.enemy_projectiles.append({
                    "pos": enemy["pos"] + np.array([12.5, 12.5]),
                    "dir": enemy_dir
                })
                enemy["shoot_timer"] = enemy["fire_rate"]
                # Sound: Enemy Shoot

        # --- Update Projectiles & Collisions ---
        player_proj_rects = []
        for p in self.player_projectiles:
            p["pos"][0] += p["dir"] * 15
            player_proj_rects.append(pygame.Rect(p["pos"][0], p["pos"][1], 8, 8))

        enemy_rects = [pygame.Rect(e["pos"][0], e["pos"][1], 25, 25) for e in self.enemies]

        for i, proj_rect in enumerate(player_proj_rects):
            collided_idx = proj_rect.collidelist(enemy_rects)
            if collided_idx != -1:
                enemy = self.enemies[collided_idx]
                enemy["health"] -= 10
                reward += 1.0
                self.score += 10
                self.player_projectiles[i] = None # Mark for removal
                # Sound: Enemy Hit
                self._create_particles(enemy["pos"] + np.array([12.5, 12.5]), 20, self.COLOR_ENEMY, 0.7)
                if enemy["health"] <= 0:
                    self.score += 50
                    self.enemies[collided_idx] = None # Mark for removal
                    # Sound: Enemy Explode
                    self._create_particles(enemy["pos"] + np.array([12.5, 12.5]), 50, self.COLOR_ENEMY, 1.0, speed=4)


        # --- Player Damage ---
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], 30, 30)
        for p in self.enemy_projectiles:
            p["pos"][0] += p["dir"] * 10
            proj_rect = pygame.Rect(p["pos"][0], p["pos"][1], 8, 8)
            if player_rect.colliderect(proj_rect):
                self.player_health -= 5
                reward -= 1.0
                p["remove"] = True
                # Sound: Player Hit
                self._create_particles(self.player_pos + np.array([15, 15]), 15, self.COLOR_PLAYER, 0.6)

        # --- Cleanup ---
        self.player_projectiles = [p for p in self.player_projectiles if p is not None and 0 < p["pos"][0] < self.LEVEL_WIDTH]
        self.enemies = [e for e in self.enemies if e is not None]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if "remove" not in p and 0 < p["pos"][0] < self.LEVEL_WIDTH]
        self.particles = [p for p in self.particles if p["life"] > 0]

        # --- Update Particles ---
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

        # --- Update Camera ---
        target_camera_x = self.player_pos[0] - self.WIDTH / 2
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        self.camera_x = max(0, min(self.LEVEL_WIDTH - self.WIDTH, self.camera_x))

        # --- Termination and Progression ---
        self.steps += 1
        terminated = False
        
        if self.player_health <= 0:
            reward = -100.0
            terminated = True
            self.game_over = True
            # Sound: Game Over

        if self.player_pos[0] > self.exit_pos[0]:
            if self.stage < self.MAX_STAGES:
                self.stage += 1
                self.player_pos = np.array([100.0, self.FLOOR_Y - 30.0])
                self._setup_stage()
                # No termination, just stage progression
            else:
                reward = 100.0
                self.score += 1000
                terminated = True
                self.game_over = True
                # Sound: Victory
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, pos, count, color, life_mult, speed=2, direction=None):
        for _ in range(count):
            if direction is None:
                angle = self.np_random.uniform(0, 2 * math.pi)
            else:
                angle = self.np_random.uniform(-0.5, 0.5) + (0 if direction > 0 else math.pi)

            vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(0.5, 1.0) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(10, 20) * life_mult,
                "max_life": 20 * life_mult,
                "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_background()
        self._render_exit()
        self._render_enemies()
        self._render_projectiles()
        self._render_player()
        self._render_particles()
        self._render_floor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Parallax background tiles
        tile_size = 80
        for i in range(-1, self.WIDTH // tile_size + 2):
            for j in range(self.HEIGHT // tile_size + 1):
                px = i * tile_size - (self.camera_x * 0.5 % tile_size)
                py = j * tile_size
                pygame.draw.rect(self.screen, self.COLOR_TILE, (px, py, tile_size - 2, tile_size - 2))

    def _render_floor(self):
        pygame.draw.rect(self.screen, self.COLOR_TILE, (0, self.FLOOR_Y, self.WIDTH, self.HEIGHT - self.FLOOR_Y))

    def _render_exit(self):
        x, y = self.exit_pos - np.array([self.camera_x, 0])
        if 0 < x < self.WIDTH:
            glow_size = math.sin(self.steps * 0.1) * 5 + 20
            pygame.gfxdraw.filled_circle(self.screen, int(x + 10), int(y + 20), int(glow_size), (*self.COLOR_EXIT, 50))
            pygame.draw.rect(self.screen, self.COLOR_EXIT, (x, y, 20, 40))
            pygame.draw.rect(self.screen, self.COLOR_WHITE, (x, y, 20, 40), 2)

    def _render_player(self):
        x, y = int(self.player_pos[0] - self.camera_x), int(self.player_pos[1])
        
        # Jetpack flame when jumping
        if not self.on_ground:
            for i in range(5):
                flame_x = x + 15 + self.np_random.uniform(-3, 3)
                flame_y = y + 30 + self.np_random.uniform(0, 10)
                flame_size = self.np_random.uniform(3, 6)
                pygame.draw.circle(self.screen, self.COLOR_PLAYER_PROJ, (flame_x, flame_y), flame_size)

        # Body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (x, y, 30, 30))
        # Eye
        eye_x = x + (20 if self.player_dir > 0 else 5)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, (eye_x, y + 10, 5, 5))

    def _render_enemies(self):
        for enemy in self.enemies:
            x, y = int(enemy["pos"][0] - self.camera_x), int(enemy["pos"][1])
            if -25 < x < self.WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_ENEMY, (x, y, 25, 25))
                # Eye
                eye_dir = -1 if self.player_pos[0] < enemy["pos"][0] else 1
                eye_x = x + (15 if eye_dir > 0 else 5)
                pygame.draw.rect(self.screen, self.COLOR_WHITE, (eye_x, y + 8, 5, 10))

    def _render_projectiles(self):
        for p in self.player_projectiles:
            x, y = int(p["pos"][0] - self.camera_x), int(p["pos"][1])
            if 0 < x < self.WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, (x, y, 8, 8))
        
        for p in self.enemy_projectiles:
            x, y = int(p["pos"][0] - self.camera_x), int(p["pos"][1])
            if 0 < x < self.WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_ENEMY_PROJ, (x, y, 8, 8))

    def _render_particles(self):
        for p in self.particles:
            x, y = int(p["pos"][0] - self.camera_x), int(p["pos"][1])
            alpha = int(255 * (p["life"] / p["max_life"]))
            if alpha > 0:
                size = int(5 * (p["life"] / p["max_life"]))
                if size > 0:
                    s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(s, (*p["color"], alpha), (size, size), size)
                    self.screen.blit(s, (x - size, y - size))

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, 200 * health_ratio, 20))

        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Stage
        stage_text = self.font_large.render(f"STAGE: {self.stage}", True, self.COLOR_WHITE)
        self.screen.blit(stage_text, (self.WIDTH/2 - stage_text.get_width()/2, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "health": self.player_health
        }

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
if __name__ == "__main__":
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy' or 'windows'

    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robot Rampage")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2 # no-op in this game
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()