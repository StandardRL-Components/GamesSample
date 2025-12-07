import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:18:19.748089
# Source Brief: brief_01016.md
# Brief Index: 1016
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Use your gravity well to capture incoming projectiles and protect the nebula nodes from destruction."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your cursor. Press space to activate the gravity well and capture projectiles."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    PLAYER_SPEED = 6.0
    GRAVITY_WELL_DURATION = 15  # steps
    GRAVITY_WELL_RADIUS = 150
    GRAVITY_WELL_COOLDOWN = 5 # steps

    # --- COLORS ---
    COLOR_BG = (10, 5, 20)
    COLOR_PLAYER = (220, 220, 255)
    COLOR_NODE = (0, 150, 255)
    COLOR_NODE_GLOW = (0, 100, 200)
    COLOR_PROJECTILE = (255, 50, 50)
    COLOR_PROJECTILE_TRAIL = (200, 40, 40)
    COLOR_GRAVITY_EFFECT = (50, 255, 50)
    COLOR_TEXT = (200, 200, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gym Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        
        # --- Persistent State (survives reset) ---
        self.total_waves_cleared = 0
        
        # --- Pre-generate static background surfaces ---
        self._starfield_surface = self._create_starfield()
        self._nebula_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)

        # --- Initialize State ---
        self.player_pos = None
        self.gravity_well_active = None
        self.gravity_well_timer = None
        self.gravity_well_cooldown_timer = None
        self.nebula_nodes = None
        self.projectiles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.current_wave = None
        self.base_gravity_strength = None
        self.base_projectile_speed = None
        self.base_num_projectiles = None
        self.base_node_health = None

        # self.reset() is called by the environment wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_wave = 1

        # Reset player
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.gravity_well_active = False
        self.gravity_well_timer = 0
        self.gravity_well_cooldown_timer = 0

        # Reset lists
        self.projectiles = []
        self.particles = []
        
        # If it's a true reset (not just a new game after a loss), reset persistent progress
        if options and options.get("full_reset", False):
            self.total_waves_cleared = 0

        self._update_progression()
        self._start_new_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        if self.game_over:
            return (
                self._get_observation(), 0, True, False, self._get_info()
            )

        # --- 1. Handle Input & Player State ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_player_input(movement, space_held)
        
        # --- 2. Update Game Logic ---
        self._update_gravity_well()
        reward += self._update_projectiles()
        self._update_particles()

        # --- 3. Check for Wave Completion ---
        if not self.projectiles and not self.game_over:
            # SFX: Wave complete fanfare
            reward += 50.0  # Wave clear bonus
            self.score += 50
            self.current_wave += 1
            self.total_waves_cleared += 1
            self._update_progression()
            self._start_new_wave()

        # --- 4. Check Termination Conditions ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _update_progression(self):
        # Gravity well pull strength increases by 0.2 every 10 successful waves.
        self.base_gravity_strength = 2.0 + (self.total_waves_cleared // 10) * 0.2
        # Projectile speed increases by 0.05 and number increases by 1 every 2 successful waves.
        progression_tier = self.current_wave // 2
        self.base_projectile_speed = 1.0 + progression_tier * 0.05
        self.base_num_projectiles = 3 + progression_tier
        # New nebula types (increased health) unlock every 5 successful waves.
        self.base_node_health = 1 + (self.total_waves_cleared // 5)

    def _start_new_wave(self):
        self.nebula_nodes = []
        node_positions = [
            pygame.Vector2(self.SCREEN_WIDTH * 0.25, self.SCREEN_HEIGHT * 0.5),
            pygame.Vector2(self.SCREEN_WIDTH * 0.75, self.SCREEN_HEIGHT * 0.5),
        ]
        for pos in node_positions:
            self.nebula_nodes.append({
                "pos": pos,
                "radius": 20,
                "health": self.base_node_health,
            })
        
        self._nebula_surface = self._create_nebula_background(self.np_random)

        for _ in range(self.base_num_projectiles):
            edge = self.np_random.integers(4)
            if edge == 0:  # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -10)
            elif edge == 1:  # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 10)
            elif edge == 2:  # Left
                pos = pygame.Vector2(-10, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            else:  # Right
                pos = pygame.Vector2(self.SCREEN_WIDTH + 10, self.np_random.uniform(0, self.SCREEN_HEIGHT))

            # Aim towards the center-ish area
            target_pos = pygame.Vector2(
                self.np_random.uniform(self.SCREEN_WIDTH * 0.3, self.SCREEN_WIDTH * 0.7),
                self.np_random.uniform(self.SCREEN_HEIGHT * 0.3, self.SCREEN_HEIGHT * 0.7),
            )
            velocity = (target_pos - pos).normalize() * self.base_projectile_speed
            
            self.projectiles.append({
                "pos": pos,
                "vel": velocity,
                "radius": 5,
                "trail": [pos.copy() for _ in range(15)]
            })

    def _handle_player_input(self, movement, space_held):
        if movement == 1: self.player_pos.y -= self.PLAYER_SPEED
        if movement == 2: self.player_pos.y += self.PLAYER_SPEED
        if movement == 3: self.player_pos.x -= self.PLAYER_SPEED
        if movement == 4: self.player_pos.x += self.PLAYER_SPEED

        self.player_pos.x = np.clip(self.player_pos.x, 0, self.SCREEN_WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.SCREEN_HEIGHT)

        if space_held and not self.gravity_well_active and self.gravity_well_cooldown_timer <= 0:
            # SFX: Gravity well activation
            self.gravity_well_active = True
            self.gravity_well_timer = self.GRAVITY_WELL_DURATION
            self.gravity_well_cooldown_timer = self.GRAVITY_WELL_DURATION + self.GRAVITY_WELL_COOLDOWN
            
    def _update_gravity_well(self):
        if self.gravity_well_timer > 0:
            self.gravity_well_timer -= 1
        else:
            self.gravity_well_active = False
        
        if self.gravity_well_cooldown_timer > 0:
            self.gravity_well_cooldown_timer -= 1

    def _update_projectiles(self):
        step_reward = 0
        projectiles_to_remove = []

        for i, p in enumerate(self.projectiles):
            # Apply gravity
            if self.gravity_well_active:
                dist_vec = self.player_pos - p["pos"]
                dist_sq = dist_vec.length_squared()
                
                if dist_sq < self.GRAVITY_WELL_RADIUS ** 2 and dist_sq > 1:
                    dist = math.sqrt(dist_sq)
                    force_mag = self.base_gravity_strength / dist
                    acceleration = dist_vec.normalize() * force_mag
                    p["vel"] += acceleration
                    
                    # Continuous reward for influencing a projectile
                    step_reward += 0.01

            # Limit max speed
            speed = p["vel"].length()
            if speed > self.PLAYER_SPEED * 1.5:
                p["vel"].scale_to_length(self.PLAYER_SPEED * 1.5)

            p["pos"] += p["vel"]
            p["trail"].append(p["pos"].copy())
            if len(p["trail"]) > 15:
                p["trail"].pop(0)

            # Check for capture
            if self.gravity_well_active and (self.player_pos - p["pos"]).length() < 10:
                # SFX: Projectile capture
                step_reward += 5.0
                projectiles_to_remove.append(i)
                self._spawn_particles(p["pos"], 30, self.COLOR_GRAVITY_EFFECT, 2.5)
                continue

            # Check for node collision
            for node in self.nebula_nodes:
                if (node["pos"] - p["pos"]).length() < node["radius"] + p["radius"]:
                    # SFX: Explosion
                    step_reward -= 20.0
                    projectiles_to_remove.append(i)
                    self.game_over = True
                    self._spawn_particles(p["pos"], 50, self.COLOR_PROJECTILE, 3.0)
                    continue
                # Proximity penalty
                elif (node["pos"] - p["pos"]).length() < node["radius"] + 50:
                    step_reward -= 0.5
            
            # Check for out of bounds
            if not self.screen.get_rect().inflate(50, 50).collidepoint(p["pos"]):
                # SFX: Projectile miss/fade out
                projectiles_to_remove.append(i)
                step_reward -= 10.0 # Miss penalty
                # No game over for misses, just a penalty
                continue

        # Remove projectiles safely
        for i in sorted(projectiles_to_remove, reverse=True):
            del self.projectiles[i]
            
        return step_reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] *= 0.98

    def _get_observation(self):
        self._render_background()
        self._render_game_objects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "projectiles_left": len(self.projectiles),
        }

    def _render_background(self):
        self.screen.blit(self._starfield_surface, (0, 0))
        self.screen.blit(self._nebula_surface, (0, 0))

    def _render_game_objects(self):
        # Render nebula nodes
        for node in self.nebula_nodes:
            pos_int = (int(node["pos"].x), int(node["pos"].y))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(node["radius"] * 1.5), self.COLOR_NODE_GLOW + (80,))
            # Main node
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], node["radius"], self.COLOR_NODE)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], node["radius"], self.COLOR_NODE)

        # Render projectiles
        for p in self.projectiles:
            # Trail
            if len(p["trail"]) > 1:
                trail_points = [(int(pt.x), int(pt.y)) for pt in p["trail"]]
                pygame.draw.lines(self.screen, self.COLOR_PROJECTILE_TRAIL, False, trail_points, 3)
            # Projectile body
            pos_int = (int(p["pos"].x), int(p["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], p["radius"], self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], p["radius"], (255, 150, 150))

        # Render particles
        for p in self.particles:
            pos_int = (int(p["pos"].x), int(p["pos"].y))
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = p["color"] + (max(0, min(255, alpha)),)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(p["radius"]), color)

        # Render player/gravity well
        player_pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        # Cursor
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (player_pos_int[0] - 5, player_pos_int[1]), (player_pos_int[0] + 5, player_pos_int[1]), 1)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (player_pos_int[0], player_pos_int[1] - 5), (player_pos_int[0], player_pos_int[1] + 5), 1)

        # Gravity well effect
        if self.gravity_well_active:
            progress = self.gravity_well_timer / self.GRAVITY_WELL_DURATION
            current_radius = int(self.GRAVITY_WELL_RADIUS * (1 - progress**2))
            alpha = int(150 * progress)
            pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], current_radius, self.COLOR_GRAVITY_EFFECT + (alpha,))
            pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], int(current_radius*0.7), self.COLOR_GRAVITY_EFFECT + (int(alpha*0.7),))
            
    def _render_ui(self):
        wave_text = self.font_ui.render(f"WAVE: {self.current_wave}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))

        proj_text = self.font_ui.render(f"PROJECTILES: {len(self.projectiles)}", True, self.COLOR_TEXT)
        self.screen.blit(proj_text, (self.SCREEN_WIDTH - proj_text.get_width() - 10, 10))

        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH/2 - score_text.get_width()/2, self.SCREEN_HEIGHT - 35))

        if self.game_over:
            game_over_text = self.font_main.render("GAME OVER", True, self.COLOR_PROJECTILE)
            self.screen.blit(game_over_text, (self.SCREEN_WIDTH/2 - game_over_text.get_width()/2, self.SCREEN_HEIGHT/2 - game_over_text.get_height()/2))
            
    def _create_starfield(self):
        surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        surface.fill(self.COLOR_BG)
        for _ in range(200):
            x = random.randint(0, self.SCREEN_WIDTH)
            y = random.randint(0, self.SCREEN_HEIGHT)
            size = random.randint(1, 2)
            brightness = random.randint(100, 200)
            pygame.draw.rect(surface, (brightness, brightness, brightness), (x, y, size, size))
        return surface

    def _create_nebula_background(self, rng):
        surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        colors = [
            (80, 20, 100, 0),  # Magenta-ish
            (20, 30, 90, 0),   # Dark Blue
            (50, 20, 80, 0),   # Purple
        ]
        for _ in range(15):
            pos = (rng.integers(0, self.SCREEN_WIDTH), rng.integers(0, self.SCREEN_HEIGHT))
            radius = rng.integers(100, 300)
            color = list(rng.choice(colors))
            
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            
            num_layers = 5
            for i in range(num_layers, 0, -1):
                layer_radius = int(radius * (i / num_layers))
                alpha = int(rng.uniform(5, 20) * (1 - (i / num_layers)))
                color[3] = alpha
                pygame.gfxdraw.filled_circle(temp_surf, radius, radius, layer_radius, tuple(color))

            surface.blit(temp_surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)
        return surface

    def _spawn_particles(self, pos, count, color, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(20, 40)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": life,
                "max_life": life,
                "radius": self.np_random.uniform(2, 5),
                "color": color
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run with the "dummy" video driver, so we unset it.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Nebula Defender")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # --- Main Game Loop ---
    while not terminated:
        # --- Action Mapping for Human Player ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- Restarting Game ---")
                obs, info = env.reset(options={"full_reset": True})
                total_reward = 0

        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()