import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:51:06.559680
# Source Brief: brief_01352.md
# Brief Index: 1352
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your coral towers from invading tentacles. Place traps and flip gravity to survive the onslaught."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor, space to place a trap, and shift to flip gravity."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 32
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_CORAL = (20, 255, 150)
    COLOR_CORAL_GLOW = (20, 255, 150, 50)
    COLOR_TENTACLE = (0, 150, 255)
    COLOR_TENTACLE_GLOW = (0, 150, 255, 60)
    COLOR_TRAP = (255, 150, 0)
    COLOR_TRAP_GLOW = (255, 150, 0, 80)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_EXPLOSION = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_GRAVITY_ARROW = (200, 200, 220, 150)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward = 0

        self.cursor_pos = [0, 0]
        self.coral_towers = []
        self.tentacles = []
        self.traps = []
        self.explosions = []
        self.particles = []

        self.gravity = 1  # 1 for down, -1 for up
        self.trap_count = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.base_spawn_interval = 200
        self.spawn_timer = 0
        self.base_tentacle_speed = 1.0
        self.tentacle_speed = 0

        # self.reset() # reset is called by the wrapper/runner

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward = 0

        # Game state
        self.gravity = 1
        self.trap_count = 10
        self.tentacle_speed = self.base_tentacle_speed
        self.spawn_timer = self.base_spawn_interval
        
        # Grid dimensions
        self.grid_w = self.SCREEN_WIDTH // self.GRID_SIZE
        self.grid_h = self.SCREEN_HEIGHT // self.GRID_SIZE
        self.cursor_pos = [self.grid_w // 2, self.grid_h // 2]

        # Reset entity lists
        self.tentacles.clear()
        self.traps.clear()
        self.explosions.clear()
        self.particles.clear()
        
        # Reset coral towers
        self.coral_towers = [
            {"pos": (self.SCREEN_WIDTH * 0.2, self.SCREEN_HEIGHT - 20), "health": 100, "radius": 20},
            {"pos": (self.SCREEN_WIDTH * 0.5, self.SCREEN_HEIGHT - 20), "health": 100, "radius": 20},
            {"pos": (self.SCREEN_WIDTH * 0.8, self.SCREEN_HEIGHT - 20), "health": 100, "radius": 20},
        ]
        
        # Button state trackers
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward = 0.1  # Survival reward

        # --- Handle Actions ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_difficulty()
        self._update_spawning()
        self._update_tentacles()
        self._update_explosions()
        self._update_particles()
        
        # --- Handle Collisions & Events ---
        self._handle_collisions()
        
        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated and not terminated: # Win condition by survival
            self.reward += 100.0
            self.game_over = True
        
        self.score += self.reward

        return (
            self._get_observation(),
            self.reward,
            terminated,
            truncated,
            self._get_info(),
        )

    # --- Private Helper Methods: Game Logic ---

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.grid_w - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.grid_h - 1)

        # --- Place Trap (on press) ---
        if space_held and not self.last_space_held:
            if self.trap_count > 0:
                pixel_pos = (
                    self.cursor_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2,
                    self.cursor_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2,
                )
                # Prevent placing on coral
                can_place = True
                for tower in self.coral_towers:
                    if math.dist(pixel_pos, tower["pos"]) < tower["radius"] + 10:
                        can_place = False
                        break
                if can_place:
                    self.traps.append({"pos": pixel_pos, "radius": 8})
                    self.trap_count -= 1
        self.last_space_held = space_held

        # --- Flip Gravity (on press) ---
        if shift_held and not self.last_shift_held:
            self.gravity *= -1
            # Visual effect for gravity flip
            for _ in range(50):
                self._create_particle(
                    (self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)),
                    (255, 255, 255),
                    count=1,
                    lifespan=20,
                    speed=5,
                )
        self.last_shift_held = shift_held

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 100 == 0:
            # Increase spawn rate (by decreasing interval)
            self.base_spawn_interval = max(30, self.base_spawn_interval * 0.99)
        if self.steps > 0 and self.steps % 200 == 0:
            # Increase tentacle speed
            self.tentacle_speed += 0.05

    def _update_spawning(self):
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            spawn_y = -20 if self.gravity == 1 else self.SCREEN_HEIGHT + 20
            spawn_x = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
            
            # Create a segmented tentacle
            num_segments = 15
            segments = [[spawn_x, spawn_y + i * 5 * -self.gravity] for i in range(num_segments)]
            
            self.tentacles.append({
                "segments": segments,
                "speed": self.tentacle_speed,
                "size": 7
            })
            self.spawn_timer = int(self.base_spawn_interval)

    def _update_tentacles(self):
        for tentacle in self.tentacles:
            # Move head
            head = tentacle["segments"][0]
            head[1] += tentacle["speed"] * self.gravity
            
            # Segments follow the previous one
            for i in range(1, len(tentacle["segments"])):
                leader = tentacle["segments"][i-1]
                follower = tentacle["segments"][i]
                dx, dy = leader[0] - follower[0], leader[1] - follower[1]
                distance = math.hypot(dx, dy)
                segment_length = 5
                if distance > segment_length:
                    angle = math.atan2(dy, dx)
                    follower[0] = leader[0] - math.cos(angle) * segment_length
                    follower[1] = leader[1] - math.sin(angle) * segment_length

    def _update_explosions(self):
        for exp in self.explosions:
            exp["timer"] -= 1
            exp["radius"] += exp["expansion_rate"]
        self.explosions = [exp for exp in self.explosions if exp["timer"] > 0]

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _handle_collisions(self):
        # Tentacles vs Coral
        for tentacle in self.tentacles[:]:
            head_pos = tentacle["segments"][0]
            for tower in self.coral_towers:
                if tower["health"] > 0 and math.dist(head_pos, tower["pos"]) < tower["radius"]:
                    tower["health"] -= 25
                    self._create_particle(head_pos, self.COLOR_CORAL, count=20, speed=3)
                    self.tentacles.remove(tentacle)
                    break 
        
        # Tentacles vs Traps
        for trap in self.traps[:]:
            for tentacle in self.tentacles[:]:
                head_pos = tentacle["segments"][0]
                if math.dist(head_pos, trap["pos"]) < trap["radius"] + tentacle["size"]:
                    self._create_explosion(trap["pos"], 60)
                    self.reward += 1.0 # Reward for destroying tentacle
                    self.tentacles.remove(tentacle)
                    self.traps.remove(trap)
                    break
        
        # Explosions vs Tentacles & other Traps
        for exp in self.explosions:
            # vs Tentacles
            for tentacle in self.tentacles[:]:
                for segment in tentacle["segments"]:
                    if math.dist(segment, exp["pos"]) < exp["radius"]:
                        self._create_particle(segment, self.COLOR_TENTACLE, count=5)
                        self.reward += 1.0 # Reward for destroying tentacle
                        self.tentacles.remove(tentacle)
                        break
            # vs Traps (chain reaction)
            for trap in self.traps[:]:
                if math.dist(trap["pos"], exp["pos"]) < exp["radius"]:
                    self._create_explosion(trap["pos"], 60)
                    self.traps.remove(trap)

        # Cleanup tentacles that go off-screen
        self.tentacles = [
            t for t in self.tentacles if (0 < t["segments"][0][1] < self.SCREEN_HEIGHT)
        ]

    def _check_termination(self):
        # Lose condition
        if all(tower["health"] <= 0 for tower in self.coral_towers):
            self.reward -= 100.0
            self.game_over = True
            return True

        return False
        
    def _create_particle(self, pos, color, count=10, lifespan=30, speed=2):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = [math.cos(angle) * speed * self.np_random.uniform(0.5, 1.5), 
                   math.sin(angle) * speed * self.np_random.uniform(0.5, 1.5)]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "color": color,
                "size": self.np_random.uniform(1, 4)
            })

    def _create_explosion(self, pos, radius, duration=15):
        self.explosions.append({
            "pos": pos,
            "max_radius": radius,
            "radius": 0,
            "timer": duration,
            "max_timer": duration,
            "expansion_rate": radius / (duration * 0.5)
        })
        self._create_particle(pos, self.COLOR_EXPLOSION, count=40, lifespan=25, speed=4)

    # --- Private Helper Methods: Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_particles()
        self._render_coral()
        self._render_traps()
        self._render_tentacles()
        self._render_explosions()
        self._render_cursor()

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = p["color"] + (alpha,)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["size"]), color
            )

    def _render_coral(self):
        for tower in self.coral_towers:
            if tower["health"] > 0:
                health_ratio = tower["health"] / 100.0
                radius = int(tower["radius"] * health_ratio)
                if radius > 0:
                    pos = (int(tower["pos"][0]), int(tower["pos"][1]))
                    # Glow
                    glow_radius = int(radius * 2.5 * (0.5 + health_ratio * 0.5))
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, self.COLOR_CORAL_GLOW)
                    # Core
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_CORAL)
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_CORAL)
    
    def _render_traps(self):
        for trap in self.traps:
            pos = (int(trap["pos"][0]), int(trap["pos"][1]))
            radius = int(trap["radius"])
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius * 2, self.COLOR_TRAP_GLOW)
            # Core
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_TRAP)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_TRAP)

    def _render_tentacles(self):
        for tentacle in self.tentacles:
            for i, segment in enumerate(tentacle["segments"]):
                pos = (int(segment[0]), int(segment[1]))
                size = int(tentacle["size"] * (1 - i / len(tentacle["segments"]) * 0.5))
                if size > 0:
                    # Glow
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size * 2, self.COLOR_TENTACLE_GLOW)
                    # Core
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_TENTACLE)

    def _render_explosions(self):
        for exp in self.explosions:
            progress = 1 - (exp["timer"] / exp["max_timer"])
            alpha = int(255 * math.sin(progress * math.pi)) # Fade in and out
            color = self.COLOR_EXPLOSION + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(exp["pos"][0]), int(exp["pos"][1]), int(exp["radius"]), color)
            pygame.gfxdraw.aacircle(self.screen, int(exp["pos"][0]), int(exp["pos"][1]), int(exp["radius"]), color)

    def _render_cursor(self):
        if self.game_over: return
        pos = (
            self.cursor_pos[0] * self.GRID_SIZE,
            self.cursor_pos[1] * self.GRID_SIZE,
        )
        rect = pygame.Rect(pos[0], pos[1], self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 1)

    def _render_ui(self):
        # Trap count
        trap_text = self.font_small.render(f"TRAPS: {self.trap_count}", True, self.COLOR_UI_TEXT)
        self.screen.blit(trap_text, (10, 10))

        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Gravity indicator
        if self.gravity == 1: # Down
            points = [(self.SCREEN_WIDTH/2 - 10, 15), (self.SCREEN_WIDTH/2 + 10, 15), (self.SCREEN_WIDTH/2, 25)]
        else: # Up
            points = [(self.SCREEN_WIDTH/2 - 10, 25), (self.SCREEN_WIDTH/2 + 10, 25), (self.SCREEN_WIDTH/2, 15)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRAVITY_ARROW)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GRAVITY_ARROW)

        # Game Over / Win message
        if self.game_over:
            if self.steps >= self.MAX_STEPS:
                msg = "SURVIVED"
            else:
                msg = "CORAL DESTROYED"
            
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "coral_health": [c["health"] for c in self.coral_towers],
            "traps_left": self.trap_count
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    truncated = False
    
    # Map keyboard keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Use a display for human play
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Coral Guardian")
    clock = pygame.time.Clock()

    while running:
        # --- Human Input ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                truncated = False

        if not terminated and not truncated:
            keys = pygame.key.get_pressed()
            for key, move_action in key_map.items():
                if keys[key]:
                    movement = move_action
                    break # Prioritize one movement key
            
            if keys[pygame.K_SPACE]:
                space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_held = 1

            action = [movement, space_held, shift_held]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            
        # --- Rendering ---
        # The observation is already the rendered image
        # We just need to convert it back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()