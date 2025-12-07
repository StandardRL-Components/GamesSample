import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
from typing import List, Dict, Any
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An arcade-style spaceship game where the player pilots a mining ship to collect ore from asteroids while dodging enemy lasers.

    The goal is to collect 100 ore without losing all 3 lives. The game features real-time action, particle effects,
    and increasing difficulty as the player's score rises.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move your ship. Hold Space to activate your mining laser on nearby asteroids. "
        "Dodge the red enemy lasers!"
    )

    game_description = (
        "Pilot a mining ship, dodging enemy lasers and collecting ore from asteroids to reach a target yield "
        "before your ship is destroyed."
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_SHIELD = (100, 255, 200, 100)
    COLOR_ASTEROID = (120, 130, 140)
    COLOR_ASTEROID_OUTLINE = (100, 110, 120)
    COLOR_LASER = (255, 50, 50)
    COLOR_LASER_CORE = (255, 255, 255)
    COLOR_ORE = (255, 220, 0)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_WHITE = (255, 255, 255)

    # Game parameters
    MAX_STEPS = 3000
    WIN_SCORE = 100
    FPS = 30

    # Player
    PLAYER_SIZE = (18, 22)
    PLAYER_ACCEL = 0.8
    PLAYER_DAMPING = 0.92
    PLAYER_MAX_LIVES = 3
    PLAYER_INVINCIBILITY_FRAMES = 90  # 3 seconds at 30fps

    # Asteroids
    MAX_ASTEROIDS = 8
    ASTEROID_SPAWN_INTERVAL = 60 # frames
    ASTEROID_MIN_SIZE = 20
    ASTEROID_MAX_SIZE = 45
    ASTEROID_MIN_ORE = 5
    ASTEROID_MAX_ORE = 25
    ASTEROID_MIN_VERTS = 7
    ASTEROID_MAX_VERTS = 12

    # Mining
    MINING_RANGE = 80
    MINING_RATE = 0.5  # Ore per frame
    
    # Lasers
    LASER_SPEED = 7.0
    LASER_BASE_HZ = 0.75
    LASER_HZ_INCREASE_PER_25_ORE = 0.2
    LASER_HZ_MAX = 2.5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player: Dict[str, Any] = {}
        self.asteroids: List[Dict[str, Any]] = []
        self.lasers: List[Dict[str, Any]] = []
        self.particles: List[Dict[str, Any]] = []
        self.stars: List[Dict[str, Any]] = []
        
        self.asteroid_spawn_timer = 0
        self.laser_spawn_timer = 0
        self.is_mining = False
        self.mining_target_asteroid = None
        
        # self.reset() is called by the wrapper/test harness, no need to call it here.
        # self.validate_implementation() # This is for debugging and not needed for the final env.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player = {
            "pos": np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32),
            "vel": np.array([0.0, 0.0], dtype=np.float32),
            "lives": self.PLAYER_MAX_LIVES,
            "invincibility_timer": 0,
            "rect": pygame.Rect(0, 0, self.PLAYER_SIZE[0], self.PLAYER_SIZE[1])
        }

        self.asteroids.clear()
        self.lasers.clear()
        self.particles.clear()
        self.stars.clear()

        for _ in range(4):
            self._spawn_asteroid()
            
        for _ in range(150):
            self._spawn_star()
            
        self.asteroid_spawn_timer = self.ASTEROID_SPAWN_INTERVAL
        self.laser_spawn_timer = self._get_laser_interval()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01  # Small penalty for existing
        ore_mined_this_step = 0
        self.is_mining = False

        # 1. Handle Input & Player Movement
        self._handle_player_movement(movement)
        
        # 2. Update Timers
        self.player["invincibility_timer"] = max(0, self.player["invincibility_timer"] - 1)
        self.asteroid_spawn_timer = max(0, self.asteroid_spawn_timer - 1)
        self.laser_spawn_timer = max(0, self.laser_spawn_timer - 1)
        
        # 3. Handle Mining
        if space_held:
            ore_mined_this_step = self._handle_mining()

        # 4. Spawners
        if self.asteroid_spawn_timer == 0 and len(self.asteroids) < self.MAX_ASTEROIDS:
            self._spawn_asteroid()
            self.asteroid_spawn_timer = self.ASTEROID_SPAWN_INTERVAL
            
        if self.laser_spawn_timer == 0:
            self._spawn_laser()
            self.laser_spawn_timer = self._get_laser_interval()

        # 5. Update Entities
        self._update_lasers()
        self._update_particles()
        
        # 6. Handle Collisions
        hit_this_frame, laser_hit = self._handle_collisions()
        if hit_this_frame:
            reward -= 5.0
            self._create_explosion(self.player["pos"], self.COLOR_PLAYER, 30)

        # 7. Update Score and Cleanup
        if ore_mined_this_step > 0:
            self.score += ore_mined_this_step
            reward += ore_mined_this_step * 0.5 # Higher reward for ore
        
        self.asteroids = [a for a in self.asteroids if a["ore"] > 0]
        self.lasers = [l for l in self.lasers if l is not laser_hit]

        # 8. Check Termination
        self.steps += 1
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            self.score = self.WIN_SCORE
            reward += 100.0
            terminated = True
            self.game_over = True
        elif self.player["lives"] <= 0:
            reward -= 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_asteroids()
        self._render_lasers()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "lives": self.player["lives"], "steps": self.steps}
    
    # --- Update Logic ---

    def _handle_player_movement(self, movement):
        acc = np.array([0.0, 0.0])
        if movement == 1: acc[1] = -self.PLAYER_ACCEL
        elif movement == 2: acc[1] = self.PLAYER_ACCEL
        elif movement == 3: acc[0] = -self.PLAYER_ACCEL
        elif movement == 4: acc[0] = self.PLAYER_ACCEL

        self.player["vel"] += acc
        self.player["vel"] *= self.PLAYER_DAMPING
        self.player["pos"] += self.player["vel"]

        # Clamp position to screen bounds
        self.player["pos"][0] = np.clip(self.player["pos"][0], self.PLAYER_SIZE[0]/2, self.SCREEN_WIDTH - self.PLAYER_SIZE[0]/2)
        self.player["pos"][1] = np.clip(self.player["pos"][1], self.PLAYER_SIZE[1]/2, self.SCREEN_HEIGHT - self.PLAYER_SIZE[1]/2)
        
        self.player["rect"].center = self.player["pos"]

    def _handle_mining(self):
        closest_asteroid = None
        min_dist = self.MINING_RANGE
        
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player["pos"] - asteroid["pos"])
            if dist < min_dist:
                min_dist = dist
                closest_asteroid = asteroid
        
        if closest_asteroid:
            self.is_mining = True
            self.mining_target_asteroid = closest_asteroid
            
            ore_to_mine = min(closest_asteroid["ore"], self.MINING_RATE)
            closest_asteroid["ore"] -= ore_to_mine
            
            if closest_asteroid["ore"] <= 0:
                self._create_explosion(closest_asteroid["pos"], self.COLOR_ASTEROID, 20)

            if self.steps % 3 == 0:
                particle_start_pos = closest_asteroid["pos"] + self.np_random.uniform(-1, 1, 2) * closest_asteroid["radius"] * 0.5
                self._spawn_particle(particle_start_pos, self.COLOR_ORE, 60, "ore")
            return ore_to_mine
            
        return 0

    def _update_lasers(self):
        for laser in self.lasers:
            laser["pos"] += laser["vel"]
        self.lasers = [l for l in self.lasers if 0 < l["pos"][0] < self.SCREEN_WIDTH]

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            p["life"] -= 1
            if p["life"] > 0:
                if p["type"] == "ore":
                    # Homing behavior
                    target_dir = self.player["pos"] - p["pos"]
                    dist = np.linalg.norm(target_dir)
                    if dist > 1:
                        target_dir /= dist
                    p["vel"] = p["vel"] * 0.8 + target_dir * 2.5
                p["pos"] += p["vel"]
                new_particles.append(p)
        self.particles = new_particles

    def _handle_collisions(self):
        if self.player["invincibility_timer"] > 0:
            return False, None
            
        for laser in self.lasers:
            if self.player["rect"].colliderect(laser["rect"]):
                self.player["lives"] -= 1
                self.player["invincibility_timer"] = self.PLAYER_INVINCIBILITY_FRAMES
                return True, laser
        return False, None

    # --- Spawners ---

    def _spawn_star(self):
        self.stars.append({
            "pos": np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)]),
            "speed": self.np_random.uniform(0.1, 0.6),
            "size": self.np_random.uniform(0.5, 1.5),
            "color": self.np_random.integers(50, 150)
        })

    def _spawn_asteroid(self):
        radius = self.np_random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)
        pos = np.array([
            self.np_random.uniform(0, self.SCREEN_WIDTH),
            self.np_random.uniform(0, self.SCREEN_HEIGHT)
        ])
        
        # Avoid spawning on player
        while np.linalg.norm(pos - self.player["pos"]) < 100:
            pos = np.array([
                self.np_random.uniform(0, self.SCREEN_WIDTH),
                self.np_random.uniform(0, self.SCREEN_HEIGHT)
            ])

        ore = self.np_random.uniform(self.ASTEROID_MIN_ORE, self.ASTEROID_MAX_ORE) * (radius / self.ASTEROID_MAX_SIZE)
        
        num_verts = self.np_random.integers(self.ASTEROID_MIN_VERTS, self.ASTEROID_MAX_VERTS + 1)
        verts = []
        for i in range(num_verts):
            angle = 2 * math.pi * i / num_verts
            r = radius * self.np_random.uniform(0.75, 1.25)
            verts.append((pos[0] + r * math.cos(angle), pos[1] + r * math.sin(angle)))

        self.asteroids.append({"pos": pos, "radius": radius, "ore": ore, "verts": verts})
    
    def _spawn_laser(self):
        side = self.np_random.choice([-1, 1])
        start_x = self.SCREEN_WIDTH if side == -1 else 0
        start_y = self.np_random.uniform(20, self.SCREEN_HEIGHT - 20)
        
        vel_x = self.LASER_SPEED * side
        
        self.lasers.append({
            "pos": np.array([start_x, start_y], dtype=np.float32),
            "vel": np.array([vel_x, 0], dtype=np.float32),
            "rect": pygame.Rect(start_x, start_y - 2, 20, 4)
        })

    def _spawn_particle(self, pos, color, life, p_type, vel=None):
        if vel is None:
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])

        self.particles.append({
            "pos": pos.copy(),
            "vel": vel,
            "life": life,
            "color": color,
            "type": p_type
        })
        
    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            self._spawn_particle(pos, color, self.np_random.integers(15, 30), "explosion")

    # --- Render Methods ---

    def _render_stars(self):
        for star in self.stars:
            star["pos"][0] -= star["speed"]
            if star["pos"][0] < 0:
                star["pos"][0] = self.SCREEN_WIDTH
            
            c = star["color"]
            pos_int = star["pos"].astype(int)
            size_int = max(1, int(star["size"]))
            pygame.draw.circle(self.screen, (c, c, c), pos_int, size_int)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            int_verts = [(int(v[0]), int(v[1])) for v in asteroid["verts"]]
            if len(int_verts) >= 3:
                pygame.gfxdraw.filled_polygon(self.screen, int_verts, self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, int_verts, self.COLOR_ASTEROID_OUTLINE)
            
    def _render_lasers(self):
        for laser in self.lasers:
            laser["rect"].center = laser["pos"]
            end_pos = laser["pos"] - laser["vel"] * 1.5
            start_pos_int = laser["pos"].astype(int)
            end_pos_int = end_pos.astype(int)
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos_int, end_pos_int, 5)
            pygame.draw.line(self.screen, self.COLOR_LASER_CORE, start_pos_int, end_pos_int, 2)

    def _render_player(self):
        pos = self.player["pos"]
        
        # Draw ship body
        ship_points = [
            (pos[0], pos[1] - self.PLAYER_SIZE[1]/2),
            (pos[0] - self.PLAYER_SIZE[0]/2, pos[1] + self.PLAYER_SIZE[1]/2),
            (pos[0] + self.PLAYER_SIZE[0]/2, pos[1] + self.PLAYER_SIZE[1]/2)
        ]
        int_ship_points = [(int(p[0]), int(p[1])) for p in ship_points]
        pygame.gfxdraw.aapolygon(self.screen, int_ship_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, int_ship_points, self.COLOR_PLAYER)
        
        # Draw engine flame
        flame_strength = np.linalg.norm(self.player["vel"])
        if flame_strength > 0.5:
            flame_len = min(15, flame_strength * 2)
            flame_width = self.PLAYER_SIZE[0] * 0.6
            flame_base_y = pos[1] + self.PLAYER_SIZE[1]/2
            flame_points = [
                (pos[0] - flame_width/2, flame_base_y),
                (pos[0] + flame_width/2, flame_base_y),
                (pos[0], flame_base_y + flame_len + self.np_random.uniform(-2, 2))
            ]
            c = (255, 150 + self.np_random.integers(0,100), 0)
            int_flame_points = [(int(p[0]), int(p[1])) for p in flame_points]
            if len(int_flame_points) >= 3:
                pygame.gfxdraw.filled_polygon(self.screen, int_flame_points, c)
                pygame.gfxdraw.aapolygon(self.screen, int_flame_points, c)

        # Draw invincibility shield
        if self.player["invincibility_timer"] > 0:
            if self.player["invincibility_timer"] % 10 < 5:
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(self.PLAYER_SIZE[1]), self.COLOR_PLAYER_SHIELD)
    
    def _render_particles(self):
        for p in self.particles:
            size_float = max(1, p["life"] / 15)
            if p["type"] == "ore":
                size_float = 2
            pos_int = p["pos"].astype(int)
            size_int = int(size_float)
            if size_int > 0:
                pygame.draw.circle(self.screen, p["color"], pos_int, size_int)
            
        # Mining beam
        if self.is_mining and self.mining_target_asteroid:
            start_pos = self.player["pos"]
            end_pos = self.mining_target_asteroid["pos"]
            alpha = 100 + self.np_random.integers(0, 100)
            color = (*self.COLOR_ORE, alpha)
            
            temp_surf = self.screen.convert_alpha()
            temp_surf.fill((0,0,0,0))
            pygame.draw.line(temp_surf, color, start_pos.astype(int), end_pos.astype(int), 3)
            self.screen.blit(temp_surf, (0,0))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"ORE: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.player["lives"]):
            x = self.SCREEN_WIDTH - 25 - (i * (self.PLAYER_SIZE[0] + 5))
            y = 20
            ship_points = [
                (x, y - self.PLAYER_SIZE[1]/2 * 0.8),
                (x - self.PLAYER_SIZE[0]/2 * 0.8, y + self.PLAYER_SIZE[1]/2 * 0.8),
                (x + self.PLAYER_SIZE[0]/2 * 0.8, y + self.PLAYER_SIZE[1]/2 * 0.8)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, ship_points)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if self.score >= self.WIN_SCORE:
            text = "MISSION COMPLETE"
            color = self.COLOR_ORE
        else:
            text = "SHIP DESTROYED"
            color = self.COLOR_LASER
            
        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        overlay.blit(text_surf, text_rect)
        self.screen.blit(overlay, (0, 0))

    # --- Helper Methods ---

    def _get_laser_interval(self):
        difficulty_tier = self.score // 25
        current_hz = min(
            self.LASER_HZ_MAX,
            self.LASER_BASE_HZ + difficulty_tier * self.LASER_HZ_INCREASE_PER_25_ORE
        )
        if current_hz <= 0: return float('inf')
        return self.FPS / current_hz

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # You must have pygame installed and a display environment.
    # To run this, you might need to comment out: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    # For headed mode, pygame.display must be initialized.
    # The environment itself is headless.
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        if keys[pygame.K_r]: # Press R to reset
            obs, info = env.reset()
            total_reward = 0
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Convert observation back to a Pygame surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()