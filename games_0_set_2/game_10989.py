import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a cosmic launcher to collect
    different colored dust particles. The core mechanic involves predicting
    trajectories and manipulating the flow of time to efficiently collect sets of
    dust, which in turn unlocks new regions of space with new challenges.
    The environment is designed with a strong emphasis on visual quality and
    satisfying game feel.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
        - up/down: Adjusts launcher angle.
        - left/right: Adjusts launcher power.
    - actions[1]: Space button (0=released, 1=held)
        - held: Speeds up time.
    - actions[2]: Shift button (0=released, 1=held)
        - held: Slows down time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a cosmic launcher to collect different colored dust particles by "
        "predicting trajectories and manipulating the flow of time."
    )
    user_guide = (
        "Use ↑↓ arrow keys to aim the launcher and ←→ to set power. "
        "Hold space to speed up time and shift to slow it down."
    )
    auto_advance = True

    # --- Persistent State (shared across resets) ---
    # Using a class-level dictionary for persistence as per the brief's requirements.
    # This state is NOT reset by the env.reset() method.
    PERSISTENT_STATE = {
        "total_dust_collected": {"red": 0, "green": 0, "blue": 0, "yellow": 0},
        "unlocked_regions": {"core"},
        "base_dust_speed": 1.0,
        "total_collections_ever": 0,
    }

    REGION_UNLOCK_CONDITIONS = {
        "azure": {"unlock_cost": {"blue": 25}},
        "verdant": {"unlock_cost": {"green": 25}},
        "crimson": {"unlock_cost": {"red": 25}},
        "golden": {"unlock_cost": {"yellow": 50, "red": 10, "green": 10, "blue": 10}},
    }
    
    DUST_COLORS_BY_REGION = {
        "core": ["red", "green", "blue"],
        "azure": ["red", "green", "blue"],
        "verdant": ["red", "green", "blue"],
        "crimson": ["red", "green", "blue"],
        "golden": ["red", "green", "blue", "yellow"],
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.GRAVITY = 0.03
        self.LAUNCHER_POS = (self.WIDTH // 2, self.HEIGHT - 20)
        self.MAX_DUST_PARTICLES = 60
        self.AUTO_LAUNCH_INTERVAL = 90  # frames

        # --- Visuals & Colors ---
        self.COLOR_BG = (10, 5, 30)
        self.COLOR_TEXT = (220, 220, 240)
        self.PARTICLE_COLORS = {
            "red": (255, 80, 80), "green": (80, 255, 80),
            "blue": (80, 180, 255), "yellow": (255, 255, 80)
        }
        self.COLOR_COLLECTOR = (255, 255, 255)
        self.COLOR_TRAJECTORY = (255, 255, 255, 100)
        self.TIME_WARP_COLORS = {
            "fast": (255, 50, 50, 15), "slow": (50, 150, 255, 15)
        }

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- Starfield Background ---
        self.starfield = [
            (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 2), random.uniform(0.1, 0.4))
            for _ in range(200)
        ]

        # --- Initialize State Variables ---
        # These are reset every episode
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.launcher_angle = -math.pi / 2
        self.launcher_power = 7.0
        self.time_warp_factor = 1.0
        self.collectors = []
        self.dust_particles = []
        self.particle_effects = []
        self.auto_launch_timer = self.AUTO_LAUNCH_INTERVAL
        
        # self.reset() # reset is called by wrapper
        # self.validate_implementation() # validation is done by tests

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset episode-specific state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Reset game elements
        self.collectors.clear()
        self.dust_particles.clear()
        self.particle_effects.clear()
        
        # Reset controls to default
        self.launcher_angle = -math.pi / 2
        self.launcher_power = 7.0
        self.time_warp_factor = 1.0
        self.auto_launch_timer = self.AUTO_LAUNCH_INTERVAL

        # Populate initial dust
        self._spawn_dust_particles(self.MAX_DUST_PARTICLES)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        # --- 1. Handle Input & Update Controls ---
        self._handle_input(action)
        
        # --- 2. Update Game Logic ---
        self.steps += 1
        self.auto_launch_timer -= 1
        if self.auto_launch_timer <= 0:
            self._launch_collector()
            self.auto_launch_timer = self.AUTO_LAUNCH_INTERVAL
            
        collection_reward = self._update_collectors()
        unlock_reward = self._check_and_process_unlocks()
        self._update_dust()
        self._update_particle_effects()
        
        # Replenish dust
        if len(self.dust_particles) < self.MAX_DUST_PARTICLES:
            self._spawn_dust_particles(self.MAX_DUST_PARTICLES - len(self.dust_particles))
        
        reward += collection_reward + unlock_reward
        self.score += reward
        
        # --- 3. Check Termination ---
        terminated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Adjust angle
        if movement == 1: self.launcher_angle -= 0.03
        if movement == 2: self.launcher_angle += 0.03
        self.launcher_angle = np.clip(self.launcher_angle, -math.pi, 0)

        # Adjust power
        if movement == 3: self.launcher_power -= 0.1
        if movement == 4: self.launcher_power += 0.1
        self.launcher_power = np.clip(self.launcher_power, 3.0, 11.0)

        # Adjust time warp
        if space_held: self.time_warp_factor += 0.05
        if shift_held: self.time_warp_factor -= 0.05
        self.time_warp_factor = np.clip(self.time_warp_factor, 0.25, 3.0)

    def _launch_collector(self):
        # sound: launch_whoosh.wav
        vel_x = self.launcher_power * math.cos(self.launcher_angle)
        vel_y = self.launcher_power * math.sin(self.launcher_angle)
        self.collectors.append({
            "pos": list(self.LAUNCHER_POS),
            "vel": [vel_x, vel_y],
            "radius": 8,
            "trail": []
        })

    def _update_collectors(self):
        reward = 0.0
        for collector in self.collectors[:]:
            collector["vel"][1] += self.GRAVITY
            collector["pos"][0] += collector["vel"][0]
            collector["pos"][1] += collector["vel"][1]

            # Add to trail
            collector["trail"].append(tuple(collector["pos"]))
            if len(collector["trail"]) > 20:
                collector["trail"].pop(0)

            # Collision with dust
            for dust in self.dust_particles[:]:
                dist = math.hypot(collector["pos"][0] - dust["pos"][0], collector["pos"][1] - dust["pos"][1])
                if dist < collector["radius"] + dust["radius"]:
                    # sound: collect_plink.wav
                    reward += 0.1
                    self.dust_particles.remove(dust)
                    GameEnv.PERSISTENT_STATE["total_dust_collected"][dust["color"]] += 1
                    GameEnv.PERSISTENT_STATE["total_collections_ever"] += 1

                    # Check for difficulty increase
                    if GameEnv.PERSISTENT_STATE["total_collections_ever"] % 500 == 0:
                        GameEnv.PERSISTENT_STATE["base_dust_speed"] += 0.05

                    # Create collection particle effect
                    for _ in range(15):
                        self.particle_effects.append(self._create_effect_particle(dust["pos"], self.PARTICLE_COLORS[dust["color"]]))
            
            # Remove if off-screen
            if not (0 < collector["pos"][0] < self.WIDTH and -50 < collector["pos"][1] < self.HEIGHT):
                self.collectors.remove(collector)
        return reward

    def _update_dust(self):
        for dust in self.dust_particles:
            dust["pos"][0] += dust["vel"][0] * self.time_warp_factor
            dust["pos"][1] += dust["vel"][1] * self.time_warp_factor
            
            # Screen wrap
            if dust["pos"][0] < 0: dust["pos"][0] = self.WIDTH
            if dust["pos"][0] > self.WIDTH: dust["pos"][0] = 0
            if dust["pos"][1] < 0: dust["pos"][1] = self.HEIGHT
            if dust["pos"][1] > self.HEIGHT: dust["pos"][1] = 0

    def _update_particle_effects(self):
        for p in self.particle_effects[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particle_effects.remove(p)

    def _spawn_dust_particles(self, count):
        available_colors = []
        for region_name in GameEnv.PERSISTENT_STATE["unlocked_regions"]:
            available_colors.extend(GameEnv.DUST_COLORS_BY_REGION.get(region_name, []))
        if not available_colors: available_colors = ["red", "green", "blue"] # Default
        
        for _ in range(count):
            color = random.choice(list(set(available_colors)))
            speed = GameEnv.PERSISTENT_STATE["base_dust_speed"]
            angle = random.uniform(0, 2 * math.pi)
            self.dust_particles.append({
                "pos": [random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT)],
                "vel": [speed * math.cos(angle), speed * math.sin(angle)],
                "color": color,
                "radius": 4
            })
    
    def _check_and_process_unlocks(self):
        reward = 0.0
        collected = GameEnv.PERSISTENT_STATE["total_dust_collected"]
        
        for region, data in self.REGION_UNLOCK_CONDITIONS.items():
            if region not in GameEnv.PERSISTENT_STATE["unlocked_regions"]:
                can_unlock = True
                for color, amount in data["unlock_cost"].items():
                    if collected.get(color, 0) < amount:
                        can_unlock = False
                        break
                if can_unlock:
                    # sound: unlock_ fanfare.wav
                    GameEnv.PERSISTENT_STATE["unlocked_regions"].add(region)
                    reward += 10.0 # Big reward for unlocking
        return reward

    def _get_observation(self):
        self._render_background()
        self._render_dust()
        self._render_collectors()
        self._render_particle_effects()
        self._render_launcher_and_trajectory()
        self._render_time_warp_effect()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for i, (x, y, r, speed) in enumerate(self.starfield):
            new_x = (x + self.steps * speed) % self.WIDTH
            color_val = int(120 + math.sin(self.steps * 0.01 + i) * 40)
            color = (max(0, color_val - 40), max(0, color_val - 40), color_val)
            pygame.gfxdraw.filled_circle(self.screen, int(new_x), int(y), r, color)

    def _render_dust(self):
        for dust in self.dust_particles:
            pos = (int(dust["pos"][0]), int(dust["pos"][1]))
            color = self.PARTICLE_COLORS[dust["color"]]
            
            # Glow effect
            glow_color = (*color, 30)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], dust["radius"] * 3, glow_color)
            
            # Core particle
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], dust["radius"], color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], dust["radius"], color)

            # Motion blur for time warp
            if self.time_warp_factor > 1.2:
                end_pos = (pos[0] - dust["vel"][0] * self.time_warp_factor * 2, pos[1] - dust["vel"][1] * self.time_warp_factor * 2)
                pygame.draw.line(self.screen, (*color, 100), pos, end_pos, 2)

    def _render_collectors(self):
        for collector in self.collectors:
            pos = (int(collector["pos"][0]), int(collector["pos"][1]))
            
            # Trail
            if len(collector["trail"]) > 1:
                for i in range(len(collector["trail"]) - 1):
                    alpha = int(255 * (i / len(collector["trail"])))
                    color = (*self.COLOR_COLLECTOR, alpha)
                    pygame.draw.line(self.screen, color, collector["trail"][i], collector["trail"][i+1], max(1, int(i/5)))

            # Collector body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], collector["radius"], self.COLOR_COLLECTOR)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], collector["radius"], self.COLOR_COLLECTOR)
            
    def _render_particle_effects(self):
        for p in self.particle_effects:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"] * (p["lifespan"] / p["max_lifespan"])), color)

    def _render_launcher_and_trajectory(self):
        # Launcher base
        pygame.gfxdraw.filled_circle(self.screen, self.LAUNCHER_POS[0], self.LAUNCHER_POS[1], 12, (80, 80, 100))
        pygame.gfxdraw.aacircle(self.screen, self.LAUNCHER_POS[0], self.LAUNCHER_POS[1], 12, (120, 120, 150))

        # Trajectory preview
        vx = self.launcher_power * math.cos(self.launcher_angle)
        vy = self.launcher_power * math.sin(self.launcher_angle)
        path = []
        px, py = self.LAUNCHER_POS
        for _ in range(50):
            vy += self.GRAVITY
            px += vx
            py += vy
            if _ % 4 == 0:
                path.append((int(px), int(py)))
        
        for point in path:
            if point[1] < self.HEIGHT:
                pygame.gfxdraw.filled_circle(self.screen, point[0], point[1], 2, self.COLOR_TRAJECTORY)

    def _render_time_warp_effect(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        if self.time_warp_factor > 1.1:
            overlay.fill(self.TIME_WARP_COLORS["fast"])
        elif self.time_warp_factor < 0.9:
            overlay.fill(self.TIME_WARP_COLORS["slow"])
        self.screen.blit(overlay, (0, 0))

    def _render_ui(self):
        # Score
        score_surf = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Dust collected
        y_offset = 40
        for color, count in GameEnv.PERSISTENT_STATE["total_dust_collected"].items():
            p_color = self.PARTICLE_COLORS[color]
            pygame.gfxdraw.filled_circle(self.screen, 20, y_offset, 6, p_color)
            pygame.gfxdraw.aacircle(self.screen, 20, y_offset, 6, p_color)
            text_surf = self.font_small.render(f"{count}", True, self.COLOR_TEXT)
            self.screen.blit(text_surf, (35, y_offset - 7))
            y_offset += 20
        
        # Unlocks
        y_offset = 10
        for region in GameEnv.PERSISTENT_STATE["unlocked_regions"]:
             text_surf = self.font_small.render(f"Region: {region.upper()}", True, self.COLOR_TEXT)
             self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, y_offset))
             y_offset += 15

        # Time Warp factor
        tw_text = f"TIME WARP: {self.time_warp_factor:.2f}x"
        tw_color = self.COLOR_TEXT
        if self.time_warp_factor > 1.1: tw_color = self.PARTICLE_COLORS["red"]
        if self.time_warp_factor < 0.9: tw_color = self.PARTICLE_COLORS["blue"]
        tw_surf = self.font_small.render(tw_text, True, tw_color)
        self.screen.blit(tw_surf, (self.WIDTH - tw_surf.get_width() - 10, self.HEIGHT - 20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_warp": self.time_warp_factor,
            "unlocked_regions": len(GameEnv.PERSISTENT_STATE["unlocked_regions"]),
            "total_dust_collected": GameEnv.PERSISTENT_STATE["total_dust_collected"]
        }
        
    def _create_effect_particle(self, pos, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        lifespan = random.randint(20, 40)
        return {
            "pos": list(pos),
            "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
            "color": color,
            "radius": random.randint(2, 4),
            "lifespan": lifespan,
            "max_lifespan": lifespan,
        }

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy video driver if you want a window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Cosmic Dust Collector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Player Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                total_reward = 0
                obs, info = env.reset()

        if terminated:
            print(f"Episode finished after {info['steps']} steps. Final score: {info['score']:.2f}")
            running = False

        # --- Rendering ---
        # The observation is already a rendered image
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()