import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An arcade puzzle game where the player launches projectiles to clear a grid of blocks.

    The goal is to clear a target number of blocks within a limited number of shots.
    Players aim their launcher and fire, with rewards given for each block destroyed
    and bonuses for efficient shots. The game emphasizes strategic aiming to maximize
    destruction with each projectile.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use ↑ and ↓ to adjust your aim. Press Space to launch a projectile."
    )

    # Short, user-facing description of the game
    game_description = (
        "Launch projectiles to strategically clear a grid of colored blocks using a limited number of shots."
    )

    # Frames only advance when an action is received.
    auto_advance = False

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    # Gameplay
    TOTAL_SHOTS = 20
    BLOCK_CLEAR_GOAL = 80
    MAX_STEPS = 1000  # Episode termination safety net
    PROJECTILE_SPEED = 12
    # Rewards
    REWARD_PER_BLOCK = 1
    REWARD_BONUS_THRESHOLD = 5
    REWARD_BONUS_AMOUNT = 5
    REWARD_WIN_BONUS = 50
    # Colors
    COLOR_BG = (25, 25, 40)
    COLOR_GRID = (45, 45, 60)
    COLOR_LAUNCHER = (230, 230, 230)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_PROJECTILE_GLOW = (255, 255, 0, 50)
    COLOR_TRAJECTORY = (0, 255, 255)
    COLOR_TEXT = (255, 255, 255)
    BLOCK_COLORS = [(255, 87, 51), (255, 195, 0), (51, 255, 87), (51, 87, 255)]
    # Grid
    GRID_ROWS = 10
    GRID_COLS = 16
    BLOCK_WIDTH = 30
    BLOCK_HEIGHT = 15
    BLOCK_SPACING = 4
    GRID_START_Y = 40

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
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game state variables are initialized in reset()
        self.game_state = "aiming"
        self.shots_remaining = 0
        self.blocks_destroyed_total = 0
        self.launch_angle = 0
        self.launcher_pos = (0, 0)
        self.blocks = []
        self.projectiles = []
        self.particles = []
        self.last_shot_blocks_destroyed = 0
        self.score = 0
        self.steps = 0
        
        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging and not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game_state = "aiming"  # 'aiming', 'firing', 'game_over'
        self.shots_remaining = self.TOTAL_SHOTS
        self.blocks_destroyed_total = 0
        self.score = 0
        self.steps = 0
        self.launch_angle = 0  # Angle in degrees from vertical (0 is straight up)
        self.launcher_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 20)
        self.projectiles.clear()
        self.particles.clear()
        self._create_block_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        truncated = False

        self._update_particles()

        if self.game_state == "firing":
            self._update_projectiles()
            if not self.projectiles:
                # Shot has finished
                reward += self.last_shot_blocks_destroyed * self.REWARD_PER_BLOCK
                if self.last_shot_blocks_destroyed >= self.REWARD_BONUS_THRESHOLD:
                    reward += self.REWARD_BONUS_AMOUNT
                
                self.score += reward
                self.game_state = "aiming"
                # Check for termination now that the turn is resolved
                if self.blocks_destroyed_total >= self.BLOCK_CLEAR_GOAL or self.shots_remaining == 0:
                    terminated = True

        elif self.game_state == "aiming":
            # Adjust angle
            if movement == 1:  # Up
                self.launch_angle -= 2
            elif movement == 2:  # Down
                self.launch_angle += 2
            self.launch_angle = np.clip(self.launch_angle, -85, 85)

            # Launch projectile
            if space_held and self.shots_remaining > 0:
                self.shots_remaining -= 1
                self._create_projectile()
                self.game_state = "firing"
                self.last_shot_blocks_destroyed = 0
                # Sound effect placeholder: sfx_launch.play()

        if terminated:
            self.game_state = "game_over"
            if self.blocks_destroyed_total >= self.BLOCK_CLEAR_GOAL:
                reward += self.REWARD_WIN_BONUS
                self.score += self.REWARD_WIN_BONUS
                # Sound effect placeholder: sfx_win.play()
            else:
                # Sound effect placeholder: sfx_lose.play()
                pass
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Per Gymnasium API, truncated episodes are also terminated
            self.game_state = "game_over"

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _create_block_grid(self):
        self.blocks.clear()
        grid_width = self.GRID_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.SCREEN_WIDTH - grid_width) // 2
        for row in range(self.GRID_ROWS):
            for col in range(self.GRID_COLS):
                x = start_x + col * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.GRID_START_Y + row * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                # FIX: self.np_random.choice on a list of tuples returns a numpy array.
                # Pygame drawing functions require tuples for colors.
                # We select a random index and then get the color tuple from the list.
                color_idx = self.np_random.integers(len(self.BLOCK_COLORS))
                color = self.BLOCK_COLORS[color_idx]
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({"rect": block_rect, "color": color})

    def _create_projectile(self):
        angle_rad = math.radians(self.launch_angle - 90) # Convert to standard angle
        vx = self.PROJECTILE_SPEED * math.cos(angle_rad)
        vy = self.PROJECTILE_SPEED * math.sin(angle_rad)
        projectile = {
            "pos": list(self.launcher_pos),
            "vel": [vx, vy],
            "radius": 6
        }
        self.projectiles.append(projectile)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj["pos"][0] += proj["vel"][0]
            proj["pos"][1] += proj["vel"][1]

            proj_rect = pygame.Rect(proj["pos"][0] - proj["radius"], proj["pos"][1] - proj["radius"], proj["radius"] * 2, proj["radius"] * 2)

            # Block collision
            hit_block = False
            for block in self.blocks[:]:
                if block["rect"].colliderect(proj_rect):
                    self._create_particles(block["rect"].center, block["color"])
                    self.blocks.remove(block)
                    self.blocks_destroyed_total += 1
                    self.last_shot_blocks_destroyed += 1
                    hit_block = True
                    # Sound effect placeholder: sfx_block_break.play()
            
            if hit_block:
                self.projectiles.remove(proj)
                continue

            # Wall collision
            if not (0 < proj["pos"][0] < self.SCREEN_WIDTH and 0 < proj["pos"][1] < self.SCREEN_HEIGHT):
                self.projectiles.remove(proj)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                "pos": list(pos),
                "vel": vel,
                "radius": self.np_random.uniform(2, 5),
                "lifespan": self.np_random.integers(20, 40),
                "color": color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "shots_remaining": self.shots_remaining,
            "blocks_destroyed": self.blocks_destroyed_total,
        }

    def _render_game(self):
        # Draw grid background
        for row in range(self.GRID_ROWS + 1):
            y = self.GRID_START_Y + row * (self.BLOCK_HEIGHT + self.BLOCK_SPACING) - self.BLOCK_SPACING / 2
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)
        grid_width = self.GRID_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.SCREEN_WIDTH - grid_width) // 2
        for col in range(self.GRID_COLS + 1):
            x = start_x + col * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING / 2
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_START_Y), (x, self.GRID_START_Y + self.GRID_ROWS * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)), 1)
        
        # Draw blocks
        for block in self.blocks:
            r = block["rect"]
            darker_color = tuple(max(0, c - 40) for c in block["color"])
            pygame.draw.rect(self.screen, darker_color, r)
            pygame.draw.rect(self.screen, block["color"], (r.x + 1, r.y + 1, r.width - 2, r.height - 2))

        # Draw launcher
        pygame.draw.polygon(self.screen, self.COLOR_LAUNCHER, [self.launcher_pos, (self.launcher_pos[0] - 8, self.launcher_pos[1] + 15), (self.launcher_pos[0] + 8, self.launcher_pos[1] + 15)])

        # Draw trajectory line
        if self.game_state == 'aiming':
            angle_rad = math.radians(self.launch_angle - 90)
            end_x = self.launcher_pos[0] + 400 * math.cos(angle_rad)
            end_y = self.launcher_pos[1] + 400 * math.sin(angle_rad)
            # Dotted line effect
            start_pos = np.array(self.launcher_pos)
            direction = np.array([math.cos(angle_rad), math.sin(angle_rad)])
            for i in range(20):
                p1 = start_pos + direction * (i * 20 + 5)
                p2 = start_pos + direction * (i * 20 + 15)
                if p1[1] < 0 or p2[1] < 0: break
                pygame.draw.line(self.screen, self.COLOR_TRAJECTORY, p1.astype(int), p2.astype(int), 1)

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(proj["radius"] * 2), self.COLOR_PROJECTILE_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(proj["radius"] * 2), self.COLOR_PROJECTILE_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], proj["radius"], self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], proj["radius"], self.COLOR_PROJECTILE)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 40))))
            color_with_alpha = p["color"] + (alpha,)
            radius = int(p["radius"])
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (radius, radius), radius)
            self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))

    def _render_ui(self):
        # Shots remaining
        shots_text = self.font_main.render(f"SHOTS: {self.shots_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(shots_text, (10, 10))

        # Blocks cleared
        blocks_text = self.font_main.render(f"CLEARED: {self.blocks_destroyed_total}/{self.BLOCK_CLEAR_GOAL}", True, self.COLOR_TEXT)
        self.screen.blit(blocks_text, (self.SCREEN_WIDTH - blocks_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_state == "game_over":
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.blocks_destroyed_total >= self.BLOCK_CLEAR_GOAL:
                msg = "YOU WIN!"
                color = (0, 255, 128)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set `human_play = False` to run a random agent test
    human_play = True

    env = GameEnv()
    
    if not human_play:
        # --- Random Agent Test ---
        print("Running random agent test...")
        obs, info = env.reset(seed=42)
        total_reward = 0
        for i in range(1000): # Run for more steps
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if (i+1) % 50 == 0:
                print(f"Step {i+1}: Reward={reward:.2f}, Info={info}")
            if terminated or truncated:
                print(f"Episode finished after {i+1} steps. Final Score: {info['score']}")
                obs, info = env.reset(seed=i) # Use a different seed for next episode
                total_reward = 0
    else:
        # --- Human Player Code ---
        print("\n" + "="*30)
        print("Human Player Mode")
        print(env.game_description)
        print(env.user_guide)
        print("="*30 + "\n")

        # Pygame setup for display
        display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Block Breaker Gym Environment")
        clock = pygame.time.Clock()
        
        obs, info = env.reset()
        running = True
        total_reward = 0
        
        while running:
            # Action defaults
            movement = 0 # none
            space = 0 # released
            shift = 0 # released
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                # Handle one-shot key presses for firing
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        space = 1
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            # space is handled by KEYDOWN event to avoid continuous fire
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift = 1
            
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the observation to the display window
            frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(frame_surface, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}")
                pygame.time.wait(2000) # Pause before reset
                obs, info = env.reset()
                total_reward = 0

            clock.tick(30) # Limit to 30 FPS

    env.close()