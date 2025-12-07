import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:39:36.345365
# Source Brief: brief_00004.md
# Brief Index: 4
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: A visually-focused Gymnasium environment where a shrinking spacecraft
    navigates a procedural asteroid field.

    Core Gameplay:
    - Player controls a ship that can move, shrink, and deploy mines.
    - The goal is to reach a target planet at a specific small size.
    - Colliding with asteroids or the planet at the wrong size is a failure.
    - The ship has a passive forward deflector that destroys small asteroids for a speed boost.
    - Deploying mines on the planet "terraforms" it and contributes to unlocking new content.
    - Difficulty increases over time with more and faster asteroids.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a procedural asteroid field in a shrinking spacecraft. "
        "Shrink to the correct size to land on the target planet while avoiding collisions."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to shrink and shift to deploy a mine."
    )
    auto_advance = True

    # --- CONFIGURATION ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_EPISODE_STEPS = 5000

    # --- COLORS ---
    COLOR_BG = pygame.Color(10, 20, 40)
    COLOR_PLAYER = pygame.Color(0, 255, 128)
    COLOR_PLAYER_GLOW = pygame.Color(0, 255, 128, 50)
    COLOR_ASTEROID = pygame.Color(255, 80, 80)
    COLOR_ASTEROID_GLOW = pygame.Color(255, 80, 80, 40)
    COLOR_MINE = pygame.Color(100, 150, 255)
    COLOR_MINE_GLOW = pygame.Color(100, 150, 255, 60)
    COLOR_PLANET = pygame.Color(180, 80, 255)
    COLOR_PLANET_GLOW = pygame.Color(180, 80, 255, 30)
    COLOR_PLANET_CRATER = pygame.Color(140, 50, 200)
    COLOR_PARTICLE_BOOST = pygame.Color(255, 150, 0)
    COLOR_PARTICLE_EXPLOSION = pygame.Color(255, 200, 150)
    COLOR_UI_TEXT = pygame.Color(220, 220, 240)
    COLOR_UI_VALUE = pygame.Color(255, 255, 255)
    COLOR_UI_COOLDOWN = pygame.Color(255, 100, 100)

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
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Persistent State (Unlocks) ---
        self._initialize_unlockable_content()
        self.unlocked_ship_indices = {0}
        self.unlocked_planet_indices = {0}

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_size = 0.0
        self.player_target_size = 0.0
        self.player_base_size = 0.0
        self.player_shrink_rate = 0.0
        self.player_shrink_cooldown = 0
        self.player_mines = 0
        self.last_movement_dir = pygame.Vector2(1, 0)

        self.asteroids = []
        self.mines = []
        self.particles = []
        self.stars = []

        self.planet_pos = pygame.Vector2(0, 0)
        self.planet_size = 0.0
        self.planet_target_size = 0.0
        self.planet_craters = []

        self.difficulty_mod = 1.0
        self.target_asteroid_count = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # validation is done by tests

    def _initialize_unlockable_content(self):
        self.ship_configs = [
            {"name": "Scout", "size": 20, "shrink_rate": 0.8, "mines": 3},
            {"name": "Interceptor", "size": 18, "shrink_rate": 0.75, "mines": 2},
            {"name": "Surveyor", "size": 25, "shrink_rate": 0.85, "mines": 5},
        ]
        self.planet_configs = [
            {"name": "Xylos", "size": 80, "target_size": 8},
            {"name": "Cryonia", "size": 100, "target_size": 10},
            {"name": "Magmar", "size": 60, "target_size": 5},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Select unlocked content ---
        ship_idx = self.np_random.choice(list(self.unlocked_ship_indices))
        planet_idx = self.np_random.choice(list(self.unlocked_planet_indices))
        ship_config = self.ship_configs[ship_idx]
        planet_config = self.planet_configs[planet_idx]

        # --- Reset episode state ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 4, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_base_size = ship_config["size"]
        self.player_size = self.player_base_size
        self.player_target_size = self.player_base_size
        self.player_shrink_rate = ship_config["shrink_rate"]
        self.player_mines = ship_config["mines"]
        self.player_shrink_cooldown = 0
        self.last_movement_dir = pygame.Vector2(1, 0)

        self.planet_pos = pygame.Vector2(self.SCREEN_WIDTH * 0.85, self.SCREEN_HEIGHT / 2)
        self.planet_size = planet_config["size"]
        self.planet_target_size = planet_config["target_size"]
        self.planet_craters = []

        self.asteroids = []
        self.mines = []
        self.particles = []
        
        self.difficulty_mod = 1.0
        self.target_asteroid_count = 5
        for _ in range(self.target_asteroid_count):
            self._spawn_asteroid()

        self.stars = [
            (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT), self.np_random.uniform(0.5, 1.5))
            for _ in range(100)
        ]
        
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1
        
        # --- 1. Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward += self._handle_input(movement, space_held, shift_held)
        
        # --- 2. Update Game State ---
        self._update_player()
        self._update_asteroids()
        self._update_mines()
        self._update_particles()
        self._update_difficulty()
        
        # --- 3. Handle Collisions & Interactions ---
        interaction_reward, terminated = self._handle_interactions()
        reward += interaction_reward
        
        # --- 4. Finalize Step ---
        self.game_over = terminated or self.steps >= self.MAX_EPISODE_STEPS
        
        # Survival reward
        if not self.game_over:
            reward += 0.01

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            self.steps >= self.MAX_EPISODE_STEPS,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0.0
        
        # Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        if move_vec.length() > 0:
            self.player_vel += move_vec.normalize() * 0.5
            self.last_movement_dir = move_vec.normalize()

        # Shrink (Spacebar) - on press
        if space_held and not self.prev_space_held and self.player_shrink_cooldown == 0:
            self.player_target_size = max(self.planet_target_size, self.player_size * self.player_shrink_rate)
            self.player_shrink_cooldown = 90 # 3 seconds cooldown
            reward += 5.0
            # sfx: shrink_activate

        # Deploy Mine (Shift) - on press
        if shift_held and not self.prev_shift_held and self.player_mines > 0:
            self.player_mines -= 1
            self._spawn_mine()
            reward += 10.0
            # sfx: mine_deploy

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return reward

    def _update_player(self):
        # Apply friction/drag
        self.player_vel *= 0.95
        
        # Clamp velocity
        if self.player_vel.length() > 5:
            self.player_vel.scale_to_length(5)
            
        # Update position and wrap around screen
        self.player_pos += self.player_vel
        self.player_pos.x %= self.SCREEN_WIDTH
        self.player_pos.y %= self.SCREEN_HEIGHT

        # Update size (interpolation)
        if abs(self.player_size - self.player_target_size) > 0.1:
            self.player_size += (self.player_target_size - self.player_size) * 0.1
        else:
            self.player_size = self.player_target_size
        
        # Update cooldowns
        self.player_shrink_cooldown = max(0, self.player_shrink_cooldown - 1)
        
        # Reset size if not shrinking
        if self.player_target_size < self.player_base_size and self.player_shrink_cooldown == 0:
             self.player_target_size = self.player_base_size

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid["pos"] += asteroid["vel"]
            asteroid["angle"] = (asteroid["angle"] + asteroid["rot_speed"]) % 360
            asteroid["pos"].x %= self.SCREEN_WIDTH
            asteroid["pos"].y %= self.SCREEN_HEIGHT

    def _update_mines(self):
        for mine in self.mines[:]:
            mine["size"] -= 0.2
            if mine["size"] <= 1:
                self.mines.remove(mine)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 500 == 0:
            self.difficulty_mod *= 1.1
            self.target_asteroid_count = int(self.target_asteroid_count * 1.1)

        # Replenish asteroids
        while len(self.asteroids) < self.target_asteroid_count:
            self._spawn_asteroid()

    def _handle_interactions(self):
        reward = 0.0
        terminated = False

        # --- Player-Asteroid Collision ---
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid["pos"])
            if dist < self.player_size + asteroid["size"]:
                self._create_explosion(self.player_pos, self.player_size)
                # sfx: player_explosion
                return -100.0, True

        # --- Player "Deflector" (Passive weapon) ---
        deflector_range = self.player_size + 40
        for asteroid in self.asteroids[:]:
            if asteroid["size"] < self.player_size * 0.8: # Can only destroy small asteroids
                dist = self.player_pos.distance_to(asteroid["pos"])
                if dist < deflector_range:
                    to_asteroid = (asteroid["pos"] - self.player_pos).normalize()
                    if self.last_movement_dir.dot(to_asteroid) > 0.8: # Is it in front?
                        self.asteroids.remove(asteroid)
                        reward += 1.0
                        self._create_explosion(asteroid["pos"], asteroid["size"])
                        self._create_boost_particles(self.player_pos)
                        self.player_vel += self.last_movement_dir * 0.5 # Speed boost
                        # sfx: asteroid_destroy_small

        # --- Player-Planet Collision ---
        dist_to_planet = self.player_pos.distance_to(self.planet_pos)
        if dist_to_planet < self.player_size + self.planet_size:
            if self.player_size <= self.planet_target_size + 2: # VICTORY
                reward += 100.0
                terminated = True
                self._unlock_content()
                # sfx: victory
            else: # FAILURE
                reward += -100.0
                terminated = True
                self._create_explosion(self.player_pos, self.player_size)
                # sfx: player_explosion_planet
        
        # --- Mine-Planet Collision ---
        for mine in self.mines[:]:
            if mine["pos"].distance_to(self.planet_pos) < mine["size"] + self.planet_size:
                self.mines.remove(mine)
                self.planet_craters.append({
                    "pos": mine["pos"],
                    "size": self.np_random.uniform(20, 40)
                })
                # sfx: terraform_impact
        
        return reward, terminated
    
    def _unlock_content(self):
        if len(self.unlocked_ship_indices) < len(self.ship_configs):
            self.unlocked_ship_indices.add(len(self.unlocked_ship_indices))
        if len(self.unlocked_planet_indices) < len(self.planet_configs):
            self.unlocked_planet_indices.add(len(self.unlocked_planet_indices))

    # --- SPAWNING METHODS ---
    def _spawn_asteroid(self):
        # Ensure asteroids don't spawn too close to the player
        while True:
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT))
            if pos.distance_to(self.player_pos) > 100:
                break
        
        size = self.np_random.uniform(5, 30)
        vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
        vel.scale_to_length(self.np_random.uniform(0.5, 1.5) * self.difficulty_mod)

        # Generate a rocky shape
        num_points = self.np_random.integers(7, 12)
        shape = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            radius = size * self.np_random.uniform(0.8, 1.2)
            shape.append(pygame.Vector2(radius * math.cos(angle), radius * math.sin(angle)))

        self.asteroids.append({
            "pos": pos,
            "vel": vel,
            "size": size,
            "angle": self.np_random.uniform(0, 360),
            "rot_speed": self.np_random.uniform(-1, 1),
            "shape": shape
        })
    
    def _spawn_mine(self):
        self.mines.append({
            "pos": self.player_pos.copy(),
            "size": self.player_base_size * 0.7,
        })
        
    def _create_explosion(self, pos, size):
        num_particles = int(size * 2)
        for _ in range(num_particles):
            vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            vel.scale_to_length(self.np_random.uniform(1, 4))
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": self.COLOR_PARTICLE_EXPLOSION,
                "size": self.np_random.uniform(1, 3)
            })
            
    def _create_boost_particles(self, pos):
        for _ in range(10):
            vel = -self.last_movement_dir * self.np_random.uniform(2, 5)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(10, 20),
                "color": self.COLOR_PARTICLE_BOOST,
                "size": self.np_random.uniform(2, 4)
            })

    # --- RENDERING ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_planet()
        self._render_asteroids()
        self._render_mines()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for x, y, size in self.stars:
            self.screen.set_at((int(x), int(y)), (100, 100, 120))

    def _render_planet(self):
        pos = (int(self.planet_pos.x), int(self.planet_pos.y))
        size = int(self.planet_size)
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + 15, self.COLOR_PLANET_GLOW)
        # Body
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_PLANET)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_PLANET)
        # Craters
        for crater in self.planet_craters:
            c_pos_rel = crater["pos"] - self.planet_pos
            c_pos = (int(self.planet_pos.x + c_pos_rel.x), int(self.planet_pos.y + c_pos_rel.y))
            pygame.gfxdraw.filled_circle(self.screen, c_pos[0], c_pos[1], int(crater["size"]), self.COLOR_PLANET_CRATER)
        
    def _render_asteroids(self):
        for asteroid in self.asteroids:
            pos = (int(asteroid["pos"].x), int(asteroid["pos"].y))
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(asteroid["size"] + 5), self.COLOR_ASTEROID_GLOW)
            
            # Body
            rotated_shape = []
            angle_rad = math.radians(asteroid["angle"])
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            for p in asteroid["shape"]:
                x = p.x * cos_a - p.y * sin_a + pos[0]
                y = p.x * sin_a + p.y * cos_a + pos[1]
                rotated_shape.append((int(x), int(y)))
            
            if len(rotated_shape) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, rotated_shape, self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, rotated_shape, self.COLOR_ASTEROID)

    def _render_mines(self):
        for mine in self.mines:
            pos = (int(mine["pos"].x), int(mine["pos"].y))
            size = int(mine["size"])
            if size > 0:
                alpha = int(max(0, min(255, (size / (self.player_base_size * 0.7)) * 255)))
                glow_color = self.COLOR_MINE_GLOW[:3] + (int(self.COLOR_MINE_GLOW.a * (alpha/255)),)
                body_color = self.COLOR_MINE[:3] + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + 5, glow_color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, body_color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, body_color)

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        size = int(self.player_size)
        
        # Glow
        for i in range(5):
            glow_size = int(size + i * 3)
            alpha = self.COLOR_PLAYER_GLOW.a - i * 8
            if alpha > 0:
                color = self.COLOR_PLAYER_GLOW[:3] + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_size, color)
        
        # Body
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_PLAYER)
        
        # Direction indicator
        if self.last_movement_dir.length() > 0:
            p2 = self.player_pos + self.last_movement_dir * (size + 5)
            pygame.draw.line(self.screen, self.COLOR_PLAYER, pos, (int(p2.x), int(p2.y)), 2)
            
    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, p["life"] * (255 / 20))
            color = p["color"][:3] + (int(alpha),)
            pygame.draw.circle(self.screen, color, (int(p["pos"].x), int(p["pos"].y)), int(p["size"]))

    def _render_ui(self):
        # Helper to draw text
        def draw_text(text, value, x, y, value_color=self.COLOR_UI_VALUE):
            text_surf = self.font_ui.render(text, True, self.COLOR_UI_TEXT)
            self.screen.blit(text_surf, (x, y))
            val_surf = self.font_ui.render(str(value), True, value_color)
            self.screen.blit(val_surf, (x + 80, y))

        draw_text("SCORE:", f"{self.score:.2f}", 10, 10)
        draw_text("SIZE:", f"{self.player_size:.1f}/{self.planet_target_size:.1f}", 10, 30)
        draw_text("MINES:", self.player_mines, 10, 50)
        
        cooldown_color = self.COLOR_UI_COOLDOWN if self.player_shrink_cooldown > 0 else self.COLOR_UI_VALUE
        cooldown_text = f"{(self.player_shrink_cooldown / self.FPS):.1f}s"
        draw_text("SHRINK:", cooldown_text, 10, 70, cooldown_color)
        
        if self.game_over:
            end_text = "VICTORY!" if self.score > 0 else "GAME OVER"
            color = self.COLOR_PLAYER if self.score > 0 else self.COLOR_ASTEROID
            text_surf = self.font_big.render(end_text, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_size": self.player_size,
            "mines_left": self.player_mines,
            "unlocked_ships": len(self.unlocked_ship_indices),
            "unlocked_planets": len(self.unlocked_planet_indices)
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block is for manual play and is not part of the Gymnasium environment API.
    # It will not be executed by the test suite.
    try:
        # --- Manual Play Example ---
        env = GameEnv()
        obs, info = env.reset()
        
        # Override Pygame display for direct rendering
        os.environ.pop("SDL_VIDEODRIVER", None)
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Shrink Racer")
        clock = pygame.time.Clock()
        
        terminated = False
        total_reward = 0
        
        print("\n--- Controls ---")
        print(GameEnv.user_guide)
        print("R: Reset")
        print("Q: Quit")
        
        while True:
            # --- Action Mapping for Manual Control ---
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    env.close()
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    terminated = True # Force reset on next loop

            if terminated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
                terminated = False

            # --- Step the environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # --- Render the observation to the display window ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(GameEnv.FPS)
    except pygame.error as e:
        print(f"Caught a Pygame error, which is expected in a headless environment: {e}")
        print("This is normal if you are not running this script with a display.")
    finally:
        pygame.quit()