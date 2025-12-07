import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:52:14.379758
# Source Brief: brief_00740.md
# Brief Index: 740
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A color-matching stealth game where a chameleon must navigate a tile-based
    environment, changing its color to blend in and avoid patrolling guards,
    racing against a ticking clock to reach the exit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A stealth game where you are a chameleon navigating a grid. Match your color to the floor tiles to avoid guards and reach the exit before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to change color and blend in. Press shift to become temporarily invisible."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_W, GRID_H = 16, 10
    TILE_SIZE = 40
    
    FPS = 30
    MAX_STEPS = 2000
    INITIAL_GAME_TIME = 90  # in seconds

    # Colors
    COLOR_BG = (20, 30, 40)
    PLAYER_COLORS = [(255, 50, 50), (50, 255, 50), (50, 50, 255)] # Red, Green, Blue
    TILE_COLORS = PLAYER_COLORS
    GUARD_COLOR = (120, 120, 140)
    GUARD_DETECT_COLOR = (255, 100, 100, 100)
    EXIT_COLOR = (255, 220, 0)
    UI_TEXT_COLOR = (220, 220, 230)
    UI_BAR_COLOR = (70, 180, 255)
    UI_BAR_BG_COLOR = (50, 50, 70)

    # Gameplay Constants
    PLAYER_SPEED = 0.2  # Interpolation speed
    GUARD_INITIAL_SPEED = 1.0 / FPS # tiles per step
    GUARD_SPEED_INCREASE_INTERVAL = 500
    GUARD_SPEED_INCREASE_AMOUNT = (0.1 / FPS)
    GUARD_DETECTION_RADIUS = 1 # 3x3 grid centered on guard
    GUARD_DETECTION_PROB = 0.1
    INVISIBILITY_DURATION = 90 # steps (3 seconds)
    INVISIBILITY_COOLDOWN = 150 # steps (5 seconds)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_visual_pos = [0.0, 0.0]
        self.player_color_idx = 0
        self.guards = []
        self.tile_grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        self.exit_pos = [0, 0]
        self.game_timer = 0
        self.invisibility_timer = 0
        self.invisibility_cooldown = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.checkpoints = []
        self.guard_speed = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player setup
        self.player_pos = [1, self.GRID_H // 2]
        self.player_visual_pos = [float(p * self.TILE_SIZE) for p in self.player_pos]
        self.player_color_idx = self.np_random.integers(0, len(self.PLAYER_COLORS))

        # Level setup
        self.exit_pos = [self.GRID_W - 2, self.GRID_H // 2]
        self.tile_grid = self.np_random.integers(0, len(self.TILE_COLORS), size=(self.GRID_W, self.GRID_H))
        # Ensure start and end tiles are not the same color initially
        if self.tile_grid[self.player_pos[0], self.player_pos[1]] == self.tile_grid[self.exit_pos[0], self.exit_pos[1]]:
             self.tile_grid[self.exit_pos[0], self.exit_pos[1]] = (self.tile_grid[self.exit_pos[0], self.exit_pos[1]] + 1) % len(self.TILE_COLORS)


        # Guard setup
        self.guards = []
        guard_paths = [
            [[3, 1], [3, self.GRID_H - 2]],
            [[self.GRID_W - 4, 1], [self.GRID_W - 4, self.GRID_H - 2]],
            [[6, 1], [self.GRID_W - 7, 1], [self.GRID_W - 7, self.GRID_H-2], [6, self.GRID_H-2]],
        ]
        for path in guard_paths:
            self.guards.append({
                "pos": list(path[0]),
                "visual_pos": [float(p * self.TILE_SIZE) for p in path[0]],
                "path": path,
                "path_idx": 0,
                "path_prog": 0.0,
            })
        self.guard_speed = self.GUARD_INITIAL_SPEED

        # Timers and state
        self.game_timer = self.INITIAL_GAME_TIME * self.FPS
        self.invisibility_timer = 0
        self.invisibility_cooldown = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        
        # Checkpoints
        self.checkpoints = []
        dist_x = self.exit_pos[0] - self.player_pos[0]
        for i in range(1, 4):
            cp_x = self.player_pos[0] + int(i * dist_x / 4)
            cp_y = self.player_pos[1]
            self.checkpoints.append([cp_x, cp_y])

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # --- Handle Actions ---
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        if space_pressed:
            # SFX: Color Change sound
            self.player_color_idx = (self.player_color_idx + 1) % len(self.PLAYER_COLORS)
            self._create_particles(self.player_visual_pos, self.PLAYER_COLORS[self.player_color_idx])

        if shift_pressed and self.invisibility_cooldown <= 0:
            # SFX: Invisibility activate sound
            self.invisibility_timer = self.INVISIBILITY_DURATION
            self.invisibility_cooldown = self.INVISIBILITY_COOLDOWN
            reward += 5.0 # Reward for using ability

        if movement != 0:
            # SFX: Move sound
            px, py = self.player_pos
            if movement == 1: py -= 1 # Up
            if movement == 2: py += 1 # Down
            if movement == 3: px -= 1 # Left
            if movement == 4: px += 1 # Right
            
            # Boundary check
            px = max(0, min(self.GRID_W - 1, px))
            py = max(0, min(self.GRID_H - 1, py))
            self.player_pos = [px, py]

        # --- Update Game State ---
        self.steps += 1
        self.game_timer -= 1
        self.invisibility_timer = max(0, self.invisibility_timer - 1)
        self.invisibility_cooldown = max(0, self.invisibility_cooldown - 1)
        
        # Update guard speed
        if self.steps > 0 and self.steps % self.GUARD_SPEED_INCREASE_INTERVAL == 0:
            self.guard_speed += self.GUARD_SPEED_INCREASE_AMOUNT

        self._update_guards()
        self._update_particles()
        
        # --- Calculate Rewards & Check Termination ---
        terminated = False
        truncated = False

        # Check for color match
        current_tile_color_idx = self.tile_grid[self.player_pos[0], self.player_pos[1]]
        is_color_matched = self.player_color_idx == current_tile_color_idx
        if is_color_matched:
            reward += 1.0

        # Check for guard detection
        is_visible = self.invisibility_timer <= 0
        if is_visible:
            for guard in self.guards:
                gx, gy = guard["pos"]
                px, py = self.player_pos
                if abs(px - gx) <= self.GUARD_DETECTION_RADIUS and abs(py - gy) <= self.GUARD_DETECTION_RADIUS:
                    if not is_color_matched:
                        reward -= 0.1 # Penalty for being visible and mismatched
                        if self.np_random.random() < self.GUARD_DETECTION_PROB:
                            # SFX: Detection alert sound
                            reward = -50.0
                            terminated = True
                            self.game_over = True
                            break
        if self.game_over: # Exit loop if detected
            pass
        # Check for reaching a checkpoint
        elif self.player_pos in self.checkpoints:
            # SFX: Checkpoint reached sound
            reward += 5.0
            self.checkpoints.remove(self.player_pos)
        # Check for win condition
        elif self.player_pos == self.exit_pos:
            # SFX: Win sound
            reward = 100.0
            terminated = True
            self.game_over = True
        # Check for loss conditions
        elif self.game_timer <= 0:
            # SFX: Time out sound
            reward = -10.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_guards(self):
        for guard in self.guards:
            guard["path_prog"] += self.guard_speed
            if guard["path_prog"] >= 1.0:
                guard["path_prog"] = 0.0
                guard["path_idx"] = (guard["path_idx"] + 1) % len(guard["path"])

            start_node = guard["path"][guard["path_idx"]]
            end_node = guard["path"][(guard["path_idx"] + 1) % len(guard["path"])]
            
            # Linear interpolation for grid position
            new_x = start_node[0] + (end_node[0] - start_node[0]) * guard["path_prog"]
            new_y = start_node[1] + (end_node[1] - start_node[1]) * guard["path_prog"]
            guard["pos"] = [round(new_x), round(new_y)]

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.2

    def _create_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.uniform(3, 6),
                'color': color,
                'life': self.np_random.integers(10, 20)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": self.game_timer / self.FPS}

    def _render_game(self):
        # Draw tiles
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                color = self.TILE_COLORS[self.tile_grid[x, y]]
                rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1) # Grid lines

        # Draw checkpoints
        for cp_pos in self.checkpoints:
            center_x = int(cp_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2)
            center_y = int(cp_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)
            pygame.draw.circle(self.screen, (255,255,255, 50), (center_x, center_y), int(self.TILE_SIZE * 0.2))

        # Draw exit
        self._draw_star(
            self.screen, self.EXIT_COLOR,
            (self.exit_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2, self.exit_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2),
            int(self.TILE_SIZE * 0.4)
        )

        # Interpolate visual positions
        target_px = self.player_pos[0] * self.TILE_SIZE
        target_py = self.player_pos[1] * self.TILE_SIZE
        self.player_visual_pos[0] += (target_px - self.player_visual_pos[0]) * self.PLAYER_SPEED
        self.player_visual_pos[1] += (target_py - self.player_visual_pos[1]) * self.PLAYER_SPEED

        for guard in self.guards:
            target_gx = guard["pos"][0] * self.TILE_SIZE
            target_gy = guard["pos"][1] * self.TILE_SIZE
            guard["visual_pos"][0] += (target_gx - guard["visual_pos"][0]) * self.PLAYER_SPEED
            guard["visual_pos"][1] += (target_gy - guard["visual_pos"][1]) * self.PLAYER_SPEED


        # Draw guards and detection radius
        for guard in self.guards:
            gx, gy = guard["pos"]
            radius = (self.GUARD_DETECTION_RADIUS * 2 + 1) * self.TILE_SIZE / 2
            detect_rect = pygame.Rect(
                (gx * self.TILE_SIZE + self.TILE_SIZE/2) - radius,
                (gy * self.TILE_SIZE + self.TILE_SIZE/2) - radius,
                radius * 2, radius * 2
            )
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            s.fill(self.GUARD_DETECT_COLOR)
            self.screen.blit(s, detect_rect.topleft)
            
            guard_center = (
                int(guard["visual_pos"][0] + self.TILE_SIZE / 2),
                int(guard["visual_pos"][1] + self.TILE_SIZE / 2)
            )
            pygame.draw.circle(self.screen, self.GUARD_COLOR, guard_center, int(self.TILE_SIZE * 0.35))
            pygame.draw.circle(self.screen, self.COLOR_BG, guard_center, int(self.TILE_SIZE * 0.35), 2)

        # Draw particles
        for p in self.particles:
            if p['radius'] > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))

        # Draw player
        player_center = (
            int(self.player_visual_pos[0] + self.TILE_SIZE / 2),
            int(self.player_visual_pos[1] + self.TILE_SIZE / 2)
        )
        player_color = self.PLAYER_COLORS[self.player_color_idx]
        self._draw_glow_circle(self.screen, player_center, int(self.TILE_SIZE * 0.4), player_color)

        # Draw invisibility shimmer
        if self.invisibility_timer > 0:
            alpha = 100 + 50 * math.sin(self.steps * 0.5) # Pulsing effect
            radius = int(self.TILE_SIZE * 0.5 * (1 - self.invisibility_timer / self.INVISIBILITY_DURATION))
            pygame.gfxdraw.filled_circle(self.screen, player_center[0], player_center[1], int(self.TILE_SIZE * 0.5), (255, 255, 255, 30))
            if self.invisibility_timer % 4 < 2: # Shimmer effect
                 pygame.gfxdraw.aacircle(self.screen, player_center[0], player_center[1], int(self.TILE_SIZE * 0.5), (255, 255, 255, int(alpha)))


    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.UI_TEXT_COLOR)
        self.screen.blit(score_text, (10, 5))

        # Timer bar
        timer_ratio = max(0, self.game_timer / (self.INITIAL_GAME_TIME * self.FPS))
        bar_width = 200
        bar_height = 18
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 8
        pygame.draw.rect(self.screen, self.UI_BAR_BG_COLOR, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.UI_BAR_COLOR, (bar_x, bar_y, int(bar_width * timer_ratio), bar_height))
        
        # Player color indicator
        indicator_x = self.SCREEN_WIDTH // 2
        indicator_y = 17
        pygame.draw.circle(self.screen, self.PLAYER_COLORS[self.player_color_idx], (indicator_x, indicator_y), 10)
        pygame.draw.circle(self.screen, self.COLOR_BG, (indicator_x, indicator_y), 10, 2)
        
        # Invisibility Cooldown indicator
        if self.invisibility_cooldown > 0:
            cooldown_ratio = self.invisibility_cooldown / self.INVISIBILITY_COOLDOWN
            pygame.draw.arc(self.screen, (200,200,200), [indicator_x-12, indicator_y-12, 24, 24], 0, cooldown_ratio * 2 * math.pi, 2)

    def _draw_glow_circle(self, surface, pos, radius, color):
        for i in range(4):
            alpha = 40 - i * 10
            glow_color = (*color, alpha)
            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, glow_color, (radius, radius), radius - i * 2)
            surface.blit(s, (pos[0] - radius, pos[1] - radius))
        pygame.draw.circle(surface, color, pos, int(radius * 0.8))

    def _draw_star(self, surface, color, center, size):
        points = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            r = size if i % 2 == 0 else size * 0.4
            points.append((center[0] + r * math.cos(angle), center[1] + r * math.sin(angle)))
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This part is for manual play and is not part of the environment's core API.
    # It's often used for debugging and visualization.
    # We need to unset the dummy video driver to see the window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Re-initialize pygame with a display
    pygame.quit()
    pygame.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Chameleon Stealth")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # For manual play, we use a different action handling loop
    last_action = [0, 0, 0]
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")

        if terminated or truncated:
            print("--- GAME OVER ---")
            print(f"Final Score: {info['score']:.2f}")
            print("Press 'R' to reset.")
            # Game is over, wait for reset
            game_over_waiting = True
            while game_over_waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game_over_waiting = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        game_over_waiting = False

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()
    pygame.quit()