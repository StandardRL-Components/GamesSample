
# Generated: 2025-08-28T02:24:58.418483
# Source Brief: brief_01697.md
# Brief Index: 1697

        
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

    user_guide = (
        "Controls: Arrow keys to move. Press Space to shoot in your last movement direction."
    )

    game_description = (
        "Survive against hordes of zombies in a top-down arena. Last for 100 seconds to win."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 3000  # 100 seconds at 30 FPS

    # Colors
    COLOR_BG = (18, 18, 18)
    COLOR_PLAYER = (0, 255, 127) # Bright Green
    COLOR_PLAYER_GLOW = (0, 255, 127, 50)
    COLOR_ZOMBIE = (220, 20, 60) # Crimson
    COLOR_BULLET = (255, 255, 0) # Yellow
    COLOR_BULLET_GLOW = (255, 255, 0, 100)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BG = (50, 50, 50)
    COLOR_HEALTH_BAR = (46, 204, 113)
    COLOR_SPLATTER = (139, 0, 0, 150)

    # Player
    PLAYER_SPEED = 4.0
    PLAYER_RADIUS = 12
    PLAYER_MAX_HEALTH = 100
    PLAYER_START_AMMO = 50

    # Zombie
    ZOMBIE_SPEED = 1.2
    ZOMBIE_RADIUS = 10
    ZOMBIE_DAMAGE = 1
    ZOMBIE_SPAWN_DIST = 150

    # Bullet
    BULLET_SPEED = 12.0
    BULLET_RADIUS = 3
    BULLET_LIFETIME = 45 # 1.5 seconds

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
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 64, bold=True)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = None
        self.player_health = None
        self.player_ammo = None
        self.player_last_move_dir = None
        self.prev_space_held = False

        self.zombies = []
        self.bullets = []
        self.particles = []

        self.zombie_spawn_timer = 0
        self.zombie_spawn_rate = 90 # Initial rate: 1 every 3 seconds
        
        self.rng = None

        self.reset()
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ammo = self.PLAYER_START_AMMO
        self.player_last_move_dir = np.array([0, -1], dtype=np.float32) # Start aiming up
        self.prev_space_held = True # Prevent shooting on first frame

        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self.zombie_spawn_rate = 90
        self.zombie_spawn_timer = self.zombie_spawn_rate

        for _ in range(2): # Initial zombies
            self._spawn_zombie()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.01 # Survival reward
        
        # --- Update Game Logic ---
        if not self.game_over:
            zombies_killed = self._update_state(movement, space_held)
            reward += zombies_killed * 1.0
            self.score += zombies_killed * 1.0

        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        if self.player_health <= 0 and not self.game_over:
            terminated = True
            self.game_over = True
            self.win = False
            reward -= 100.0
            self.score -= 100.0
            # sfx: player_death

        if self.steps >= self.MAX_STEPS and not self.game_over:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100.0
            self.score += 100.0
            # sfx: victory
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _update_state(self, movement, space_held):
        # 1. Handle player input and create bullets
        self._handle_player_input(movement, space_held)

        # 2. Update bullets and check collisions with zombies
        zombies_killed = self._update_bullets()

        # 3. Update zombies and check collisions with player
        self._update_zombies()

        # 4. Update particles
        self._update_particles()

        # 5. Handle spawning
        self._handle_spawning()
        
        return zombies_killed

    def _handle_player_input(self, movement, space_held):
        # Player Movement
        move_dir = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_dir[1] = -1 # Up
        elif movement == 2: move_dir[1] = 1 # Down
        elif movement == 3: move_dir[0] = -1 # Left
        elif movement == 4: move_dir[0] = 1 # Right

        if np.any(move_dir):
            # Normalize for consistent speed
            norm = np.linalg.norm(move_dir)
            self.player_pos += move_dir / norm * self.PLAYER_SPEED
            self.player_last_move_dir = move_dir / norm
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

        # Player Shooting
        if space_held and not self.prev_space_held and self.player_ammo > 0:
            self.player_ammo -= 1
            bullet_pos = self.player_pos + self.player_last_move_dir * self.PLAYER_RADIUS
            self.bullets.append({
                "pos": bullet_pos,
                "vel": self.player_last_move_dir * self.BULLET_SPEED,
                "life": self.BULLET_LIFETIME
            })
            # sfx: shoot
            self._create_muzzle_flash()
        
        self.prev_space_held = space_held

    def _update_bullets(self):
        zombies_killed_this_step = 0
        self.bullets = [b for b in self.bullets if b["life"] > 0]
        
        for bullet in self.bullets:
            bullet["pos"] += bullet["vel"]
            bullet["life"] -= 1

            for zombie in self.zombies:
                if np.linalg.norm(bullet["pos"] - zombie["pos"]) < self.BULLET_RADIUS + self.ZOMBIE_RADIUS:
                    zombie["health"] = 0
                    bullet["life"] = 0 # Mark for removal
                    zombies_killed_this_step += 1
                    # sfx: zombie_hit
                    self._create_blood_splatter(zombie["pos"])
                    break # Bullet can only hit one zombie
        
        self.zombies = [z for z in self.zombies if z["health"] > 0]
        return zombies_killed_this_step

    def _update_zombies(self):
        for zombie in self.zombies:
            direction = self.player_pos - zombie["pos"]
            norm = np.linalg.norm(direction)
            if norm > 1: # Avoid division by zero
                zombie["pos"] += direction / norm * self.ZOMBIE_SPEED
            
            # Collision with player
            if np.linalg.norm(self.player_pos - zombie["pos"]) < self.PLAYER_RADIUS + self.ZOMBIE_RADIUS:
                self.player_health = max(0, self.player_health - self.ZOMBIE_DAMAGE)
                # sfx: player_hit
                # Add a small knockback/particle effect for feedback
                if self.rng.random() < 0.2:
                    self._create_hit_spark(self.player_pos)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if "radius_decay" in p:
                p["radius"] = max(0, p["radius"] - p["radius_decay"])

    def _handle_spawning(self):
        # Difficulty scaling
        if self.steps > 0 and self.steps % 600 == 0:
            self.zombie_spawn_rate = max(20, self.zombie_spawn_rate - 10) # 1 zombie/60 steps -> 1 zombie/2sec

        self.zombie_spawn_timer += 1
        if self.zombie_spawn_timer >= self.zombie_spawn_rate:
            self.zombie_spawn_timer = 0
            self._spawn_zombie()

    def _spawn_zombie(self):
        while True:
            pos = self.rng.uniform([0, 0], [self.WIDTH, self.HEIGHT])
            if np.linalg.norm(pos - self.player_pos) > self.ZOMBIE_SPAWN_DIST:
                self.zombies.append({"pos": pos.astype(np.float32), "health": 1})
                break

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player_health, "ammo": self.player_ammo}

    def _render_game(self):
        # Render particles
        for p in self.particles:
            if p["type"] == "splatter":
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), self.COLOR_SPLATTER)
            elif p["type"] == "muzzle":
                self._draw_star(p["pos"], int(p["radius"]), p["color"])
            elif p["type"] == "spark":
                pygame.draw.line(self.screen, p["color"], p["pos"], p["pos"] + p["vel"] * 3, int(p["radius"]))
        
        # Render zombies
        for zombie in self.zombies:
            x, y = int(zombie["pos"][0]), int(zombie["pos"][1])
            r = self.ZOMBIE_RADIUS
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, (x - r, y - r, 2 * r, 2 * r))

        # Render bullets
        for bullet in self.bullets:
            x, y = int(bullet["pos"][0]), int(bullet["pos"][1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BULLET_RADIUS, self.COLOR_BULLET)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BULLET_RADIUS + 2, self.COLOR_BULLET_GLOW)

        # Render player
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS + 4, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        
        # Aiming indicator
        aim_end = self.player_pos + self.player_last_move_dir * (self.PLAYER_RADIUS + 5)
        pygame.draw.line(self.screen, (255,255,255), (px, py), (int(aim_end[0]), int(aim_end[1])), 2)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), bar_height))
        
        # Ammo Count
        ammo_text = self.font_small.render(f"AMMO: {self.player_ammo}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (15, 35))
        
        # Timer
        time_text = self.font_small.render(f"TIME: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 15, 10))
        
        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU SURVIVED" if self.win else "GAME OVER"
            end_text = self.font_large.render(message, True, self.COLOR_PLAYER if self.win else self.COLOR_ZOMBIE)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    # --- Particle Effects ---
    def _create_muzzle_flash(self):
        pos = self.player_pos + self.player_last_move_dir * (self.PLAYER_RADIUS + 5)
        self.particles.append({
            "pos": pos, "vel": np.array([0,0]), "life": 4, "radius": 20, 
            "color": (255, 220, 100, 200), "type": "muzzle"
        })

    def _create_blood_splatter(self, pos):
        for _ in range(self.rng.integers(10, 15)):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "life": self.rng.integers(15, 30),
                "radius": self.rng.uniform(3, 8), "radius_decay": 0.2, "type": "splatter"
            })

    def _create_hit_spark(self, pos):
        angle = self.rng.uniform(0, 2 * math.pi)
        speed = self.rng.uniform(2, 5)
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed
        self.particles.append({
            "pos": pos.copy(), "vel": vel, "life": 8, "radius": 3,
            "color": (255, 80, 80), "type": "spark"
        })

    def _draw_star(self, pos, radius, color):
        points = []
        for i in range(10):
            angle = i * math.pi / 5
            r = radius if i % 2 == 0 else radius / 2
            points.append((pos[0] + r * math.cos(angle), pos[1] + r * math.sin(angle)))
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Zombie Arena")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    pygame.quit()