import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold space to fire your mining laser at the nearest asteroid."
    )

    game_description = (
        "Pilot a mining ship through an asteroid field. Extract resources by firing your laser, but avoid collisions."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.W, self.H = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (50, 255, 50, 50)
        self.COLOR_ASTEROID = (150, 150, 150)
        self.COLOR_ORE = (255, 255, 0)
        self.COLOR_BEAM = (255, 255, 200)
        self.COLOR_THRUST = (255, 150, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_BG = (0, 0, 0, 128)
        self.COLOR_HEALTH_GOOD = (0, 200, 0)
        self.COLOR_HEALTH_BAD = (200, 0, 0)

        # --- Fonts ---
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_title = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_title = pygame.font.SysFont(None, 30)

        # --- Game Constants ---
        self.MAX_STEPS = 1000
        self.WIN_ORE = 100
        self.PLAYER_MAX_HEALTH = 5
        self.PLAYER_ACCEL = 0.2
        self.PLAYER_MAX_SPEED = 4.0
        self.PLAYER_TURN_SPEED = 0.1
        self.PLAYER_DRAG = 0.98
        self.PLAYER_BRAKE_DRAG = 0.9
        self.PLAYER_INVINCIBILITY_FRAMES = 60 # 2 seconds at 30fps
        self.ASTEROID_SIZES = {
            "small": {"radius": 12, "ore": 1, "score": 1},
            "medium": {"radius": 20, "ore": 3, "score": 3},
            "large": {"radius": 30, "ore": 5, "score": 5},
        }
        self.MINING_RANGE = 200
        self.MINING_ANGLE = math.pi / 4 # 45 degree cone
        self.MINING_RATE = 0.1 # ore per step

        # Initialize state variables
        self.player = {}
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles.clear()
        
        self._init_player()
        self._init_stars()
        self._init_asteroids()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, _ = action
        
        reward = 0
        
        # --- Update Game Logic ---
        mining_reward = self._update_player(movement, space_held == 1)
        collision_penalty = self._handle_collisions()
        self._update_asteroids()
        self._update_particles()
        
        # Ensure there are always asteroids to mine
        if not self.asteroids:
            self._spawn_asteroids(self.np_random.integers(3, 6))

        reward += mining_reward
        reward += collision_penalty
        
        # --- Termination and Score ---
        self.steps += 1
        terminated = False
        
        if self.player["health"] <= 0:
            terminated = True
            reward += -50.0
            self._create_explosion(self.player["pos"], 50, self.COLOR_PLAYER)
        
        if self.player["ore"] >= self.WIN_ORE:
            self.player["ore"] = self.WIN_ORE # Clamp
            terminated = True
            reward += 100.0
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Gymnasium standard
        
        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_background()
        self._render_asteroids()
        self._render_particles()
        self._render_player()
        
        if self.game_over:
            self._render_game_over()
        else:
            self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ore": self.player.get("ore", 0),
            "health": self.player.get("health", 0),
        }

    # --- INITIALIZATION HELPERS ---
    
    def _init_player(self):
        self.player = {
            "pos": pygame.math.Vector2(self.W / 2, self.H / 2),
            "vel": pygame.math.Vector2(0, 0),
            "angle": -math.pi / 2,
            "health": self.PLAYER_MAX_HEALTH,
            "ore": 0,
            "invincibility": 0,
            "radius": 10
        }

    def _init_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                "pos": pygame.math.Vector2(self.np_random.uniform(0, self.W), self.np_random.uniform(0, self.H)),
                "depth": self.np_random.uniform(0.1, 0.6),
                "brightness": self.np_random.integers(50, 150)
            })

    def _init_asteroids(self):
        self.asteroids.clear()
        self._spawn_asteroids(12)

    def _spawn_asteroids(self, num):
        for _ in range(num):
            size_name = self.np_random.choice(list(self.ASTEROID_SIZES.keys()))
            props = self.ASTEROID_SIZES[size_name]
            
            # Ensure asteroids don't spawn too close to the player
            while True:
                pos = pygame.math.Vector2(
                    self.np_random.uniform(0, self.W), self.np_random.uniform(0, self.H)
                )
                if pos.distance_to(self.player["pos"]) > props["radius"] + self.player["radius"] + 50:
                    break

            self.asteroids.append(self._create_asteroid(pos, size_name))
            
    def _create_asteroid(self, pos, size_name):
        props = self.ASTEROID_SIZES[size_name]
        num_vertices = self.np_random.integers(7, 12)
        base_points = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            radius = props["radius"] * self.np_random.uniform(0.7, 1.1)
            base_points.append(pygame.math.Vector2(radius, 0).rotate_rad(angle))
            
        return {
            "pos": pos,
            "size_name": size_name,
            "radius": props["radius"],
            "ore": props["ore"],
            "score": props["score"],
            "angle": self.np_random.uniform(0, 2 * math.pi),
            "rot_speed": self.np_random.uniform(-0.02, 0.02),
            "base_points": base_points,
            "health": props["ore"]
        }

    # --- UPDATE LOGIC ---

    def _update_player(self, movement, space_held):
        # Handle turning
        if movement == 3: # Left
            self.player["angle"] -= self.PLAYER_TURN_SPEED
        if movement == 4: # Right
            self.player["angle"] += self.PLAYER_TURN_SPEED

        # Handle thrust and braking
        if movement == 1: # Up
            accel = pygame.math.Vector2(self.PLAYER_ACCEL, 0).rotate_rad(self.player["angle"])
            self.player["vel"] += accel
            self._create_thrust_particles()
        elif movement == 2: # Down
            self.player["vel"] *= self.PLAYER_BRAKE_DRAG
        else: # Drag
            self.player["vel"] *= self.PLAYER_DRAG
            
        # Cap speed
        if self.player["vel"].length() > self.PLAYER_MAX_SPEED:
            self.player["vel"].scale_to_length(self.PLAYER_MAX_SPEED)

        # Update position and wrap around screen
        self.player["pos"] += self.player["vel"]
        self.player["pos"].x %= self.W
        self.player["pos"].y %= self.H
        
        # Update invincibility
        if self.player["invincibility"] > 0:
            self.player["invincibility"] -= 1

        # Handle mining
        mining_reward = 0
        if space_held:
            mining_reward = self._handle_mining()
        return mining_reward

    def _handle_mining(self):
        # Find target
        target = None
        min_dist = self.MINING_RANGE
        for a in self.asteroids:
            dist = self.player["pos"].distance_to(a["pos"])
            if dist < min_dist:
                player_to_asteroid = (a["pos"] - self.player["pos"]).normalize()
                forward_vec = pygame.math.Vector2(1, 0).rotate_rad(self.player["angle"])
                angle_diff = player_to_asteroid.angle_to(forward_vec)
                
                if abs(angle_diff) < math.degrees(self.MINING_ANGLE):
                    min_dist = dist
                    target = a
        
        reward = 0
        if target:
            # Create beam effect
            self.particles.append({
                "type": "beam", "start": self.player["pos"].copy(), "end": target["pos"].copy(),
                "life": 2, "width": self.np_random.integers(2, 5)
            })
            # Create spark effect
            if self.np_random.random() < 0.8:
                self.particles.append({
                    "type": "spark", "pos": target["pos"].copy(), "life": 10,
                    "vel": pygame.math.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
                    "radius": self.np_random.uniform(1, 3)
                })

            # Mine the asteroid
            mined_amount = self.MINING_RATE
            if target["health"] > 0:
                target["health"] -= mined_amount
                self.player["ore"] += mined_amount
                reward += mined_amount * 0.1 # Continuous reward for mining
                
                if target["health"] <= 0:
                    reward += target["score"] # Event reward for destroying
                    self._create_explosion(target["pos"], int(target["radius"]), self.COLOR_ASTEROID)
                    self._create_ore_particles(target["pos"], int(target["ore"]))
                    self.asteroids.remove(target)
                    self._spawn_asteroids(1) # Respawn one
        return reward

    def _handle_collisions(self):
        if self.player["invincibility"] > 0:
            return 0

        penalty = 0
        for asteroid in self.asteroids:
            dist = self.player["pos"].distance_to(asteroid["pos"])
            if dist < self.player["radius"] + asteroid["radius"]:
                self.player["health"] -= 1
                self.player["invincibility"] = self.PLAYER_INVINCIBILITY_FRAMES
                penalty -= 0.5
                
                # Knockback
                knockback_vec = (self.player["pos"] - asteroid["pos"]).normalize()
                self.player["vel"] += knockback_vec * 2
                
                self._create_explosion(self.player["pos"], 20, self.COLOR_THRUST)
                
                # No need to check more asteroids this frame
                return penalty
        return 0

    def _update_asteroids(self):
        for a in self.asteroids:
            a["angle"] += a["rot_speed"]

    def _update_particles(self):
        for p in self.particles[:]:
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
                continue
            
            if p["type"] in ["spark", "thrust", "explosion", "ore"]:
                p["pos"] += p["vel"]
            if p["type"] == "ore":
                # Home in on player
                direction = (self.player["pos"] - p["pos"]).normalize()
                p["vel"] = p["vel"] * 0.9 + direction * 0.5

    # --- PARTICLE CREATORS ---
    
    def _create_thrust_particles(self):
        if self.np_random.random() < 0.7:
            angle = self.player["angle"] + math.pi + self.np_random.uniform(-0.3, 0.3)
            vel = pygame.math.Vector2(2, 0).rotate_rad(angle)
            pos = self.player["pos"] - pygame.math.Vector2(self.player["radius"], 0).rotate_rad(self.player["angle"])
            self.particles.append({
                "type": "thrust", "pos": pos, "vel": vel,
                "life": self.np_random.integers(10, 20), "radius": self.np_random.uniform(1, 4)
            })

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.math.Vector2(speed, 0).rotate_rad(angle)
            self.particles.append({
                "type": "explosion", "pos": pos.copy(), "vel": vel, "color": color,
                "life": self.np_random.integers(20, 40), "radius": self.np_random.uniform(1, 3)
            })
            
    def _create_ore_particles(self, pos, num):
        for _ in range(num):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2)
            vel = pygame.math.Vector2(speed, 0).rotate_rad(angle)
            self.particles.append({
                "type": "ore", "pos": pos.copy(), "vel": vel,
                "life": self.np_random.integers(60, 100), "radius": self.np_random.uniform(2, 4)
            })

    # --- RENDER HELPERS ---

    def _render_background(self):
        player_vel_normalized = self.player["vel"].copy()
        if player_vel_normalized.length() > 0:
            player_vel_normalized.normalize_ip()

        for star in self.stars:
            pos = star["pos"] - player_vel_normalized * star["depth"] * 5
            pos.x %= self.W
            pos.y %= self.H
            color_val = star["brightness"]
            if self.np_random.random() < 0.005: # Twinkle
                color_val = 255
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (int(pos.x), int(pos.y)), 1)
            
    def _render_asteroids(self):
        for a in self.asteroids:
            points = [p.rotate_rad(a["angle"]) + a["pos"] for p in a["base_points"]]
            int_points = [(int(p.x), int(p.y)) for p in points]
            if len(int_points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_ASTEROID)

    def _render_player(self):
        if self.player["health"] <= 0: return

        # Flicker when invincible
        if self.player["invincibility"] > 0 and self.steps % 6 < 3:
            return

        angle_deg = math.degrees(self.player["angle"])
        
        # Ship body
        p1 = self.player["pos"] + pygame.math.Vector2(self.player["radius"], 0).rotate(angle_deg)
        p2 = self.player["pos"] + pygame.math.Vector2(-self.player["radius"] / 2, self.player["radius"] * 0.8).rotate(angle_deg)
        p3 = self.player["pos"] + pygame.math.Vector2(-self.player["radius"] / 2, -self.player["radius"] * 0.8).rotate(angle_deg)
        points = [(int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (int(p3.x), int(p3.y))]
        
        # Glow effect
        pygame.gfxdraw.filled_trigon(self.screen, *points[0], *points[1], *points[2], self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aatrigon(self.screen, *points[0], *points[1], *points[2], self.COLOR_PLAYER_GLOW)
        
        # Main ship
        pygame.gfxdraw.filled_trigon(self.screen, *points[0], *points[1], *points[2], self.COLOR_PLAYER)
        pygame.gfxdraw.aatrigon(self.screen, *points[0], *points[1], *points[2], self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            if p["type"] == "beam":
                pygame.draw.aaline(self.screen, self.COLOR_BEAM, p["start"], p["end"], p["width"])
            else:
                # All other particle types have a 'pos' key.
                pos = (int(p["pos"].x), int(p["pos"].y))
                if p["type"] == "thrust":
                    alpha = p["life"] / 20
                    color = (int(self.COLOR_THRUST[0]), int(self.COLOR_THRUST[1]), int(self.COLOR_THRUST[2]), int(alpha * 255))
                    pygame.gfxdraw.filled_circle(self.screen, *pos, int(p["radius"] * alpha), color)
                elif p["type"] == "spark":
                    pygame.draw.circle(self.screen, self.COLOR_BEAM, pos, int(p["radius"]))
                elif p["type"] == "explosion":
                    alpha = p["life"] / 40
                    color = (p["color"][0], p["color"][1], p["color"][2], int(alpha * 200))
                    pygame.gfxdraw.filled_circle(self.screen, *pos, int(p["radius"] * (1.5 - alpha)), color)
                elif p["type"] == "ore":
                    pygame.gfxdraw.filled_circle(self.screen, *pos, int(p["radius"]), self.COLOR_ORE)
                    pygame.gfxdraw.aacircle(self.screen, *pos, int(p["radius"]), self.COLOR_ORE)


    def _render_ui(self):
        ui_surface = pygame.Surface((self.W, 40), pygame.SRCALPHA)
        ui_surface.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surface, (0, 0))

        # Ore
        ore_text = self.font_ui.render(f"ORE: {int(self.player['ore'])}/{self.WIN_ORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ore_text, (10, 8))
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.W - score_text.get_width() - 10, 8))

        # Health
        health_text = self.font_ui.render("HEALTH:", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (self.W/2 - 100, 8))
        health_ratio = max(0, self.player["health"] / self.PLAYER_MAX_HEALTH)
        health_bar_width = 120
        health_bar_fill = int(health_bar_width * health_ratio)
        health_color = [g + (b-g) * (1-health_ratio) for g, b in zip(self.COLOR_HEALTH_GOOD, self.COLOR_HEALTH_BAD)]
        
        pygame.draw.rect(self.screen, (50, 50, 50), (self.W/2 - 10, 10, health_bar_width, 20))
        if health_bar_fill > 0:
            pygame.draw.rect(self.screen, health_color, (self.W/2 - 10, 10, health_bar_fill, 20))

    def _render_game_over(self):
        s = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        s.fill((0,0,0,180))
        self.screen.blit(s, (0,0))
        
        msg = "MISSION COMPLETE" if self.player['ore'] >= self.WIN_ORE else "SHIP DESTROYED"
        text = self.font_title.render(msg, True, self.COLOR_UI_TEXT)
        text_rect = text.get_rect(center=(self.W/2, self.H/2 - 20))
        self.screen.blit(text, text_rect)

        score_text = self.font_ui.render(f"Final Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(center=(self.W/2, self.H/2 + 20))
        self.screen.blit(score_text, score_rect)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    # It will not run in a headless environment
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass

    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Asteroid Miner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()