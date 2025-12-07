import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent stacks falling blocks.

    The goal is to stack 20 blocks. The agent controls the tilt of a platform
    to influence the horizontal position of the falling block. Each successfully
    placed block increases the 'momentum', making the tilt control more sensitive.
    Matching the color of the new block with the one below it provides temporary
    stability, freezing the two blocks together. The episode ends if the stack
    collapses, 20 blocks are stacked, or 1000 steps are reached.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]` (Movement): 0=None, 3=Tilt Left, 4=Tilt Right (1, 2 are unused)
    - `action[1]` (Space): Unused
    - `action[2]` (Shift): Unused

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - +100 for winning (stacking 20 blocks).
    - -100 for the stack collapsing.
    - +0.1 for each successfully placed block.
    - +1.0 for matching colors, granting stability.
    - -0.01 for each tilt action.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Stack falling blocks on a tilting platform. Match block colors for temporary stability and aim to build a tower 20 blocks high without it collapsing."
    user_guide = "Controls: ←→ to tilt the platform left and right."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30

    # Colors
    COLOR_BG_TOP = (15, 20, 35)
    COLOR_BG_BOTTOM = (5, 5, 10)
    COLOR_PLATFORM_BASE = (50, 50, 60)
    COLOR_PLATFORM_TOP = (220, 220, 255)
    COLOR_UI_TEXT = (230, 230, 230)
    BLOCK_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    BLOCK_OUTLINE_DARKEN = 0.6

    # Game Parameters
    PLATFORM_Y = 350
    PLATFORM_WIDTH = 250
    GROUND_Y = 370
    GRAVITY = 0.4
    MAX_TILT_ANGLE = 15  # degrees
    TILT_INTERPOLATION = 0.15
    WIN_STACK_HEIGHT = 20
    MAX_EPISODE_STEPS = 1000
    MOMENTUM_PER_BLOCK = 0.02 # Multiplier for tilt force
    STABILITY_TIMER_FRAMES = 60 # How long color-matched blocks stay frozen
    TOPPLE_FORCE = 0.0005 # Angular acceleration for unstable blocks
    ANGULAR_DAMPING = 0.95
    COLLAPSE_ANGLE_THRESHOLD = 45 # degrees

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
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        self.render_mode = render_mode
        
        # Initialize state variables to be defined in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.platform_tilt = 0.0
        self.target_tilt = 0.0
        self.momentum = 0.0
        self.stacked_blocks = []
        self.falling_block = None
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.platform_tilt = 0.0
        self.target_tilt = 0.0
        self.momentum = 0.0
        
        self.stacked_blocks = []
        self.particles = []
        
        self._spawn_falling_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0

        # 1. Update Player Controls
        if movement == 3:  # Left
            self.target_tilt = -self.MAX_TILT_ANGLE
            reward -= 0.01
        elif movement == 4:  # Right
            self.target_tilt = self.MAX_TILT_ANGLE
            reward -= 0.01
        else:  # None
            self.target_tilt = 0

        self.platform_tilt += (self.target_tilt - self.platform_tilt) * self.TILT_INTERPOLATION

        # 2. Update Game State
        landed_this_step, color_match = self._update_falling_block()
        if landed_this_step:
            reward += 0.1
            if color_match:
                reward += 1.0

        collapse_detected = self._update_stack_stability()
        self._update_particles()
        
        self.steps += 1

        # 3. Check Termination Conditions
        terminated = False
        truncated = False
        if collapse_detected:
            terminated = True
            self.game_over = True
            reward = -100.0
        elif len(self.stacked_blocks) >= self.WIN_STACK_HEIGHT:
            terminated = True
            self.game_over = True
            reward = 100.0
        
        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _spawn_falling_block(self):
        w = self.np_random.choice([60, 80, 100])
        h = self.np_random.choice([20, 25])
        x = self.SCREEN_WIDTH / 2
        y = h # Start just off-screen
        color_idx = self.np_random.integers(len(self.BLOCK_COLORS))
        color = self.BLOCK_COLORS[color_idx]
        
        self.falling_block = {
            "pos": np.array([x, y], dtype=float),
            "vel": np.array([0.0, 0.0], dtype=float),
            "size": (w, h),
            "color": color,
            "angle": 0.0
        }

    def _update_falling_block(self):
        if not self.falling_block:
            return False, False

        # Apply tilt force (scaled by momentum)
        tilt_rad = math.radians(self.platform_tilt)
        force_x = math.sin(tilt_rad) * (1.0 + self.momentum)
        self.falling_block["vel"][0] += force_x
        
        # Apply gravity
        self.falling_block["vel"][1] += self.GRAVITY
        
        # Apply velocity
        self.falling_block["pos"] += self.falling_block["vel"]

        # Damp horizontal velocity
        self.falling_block["vel"][0] *= 0.98

        # Check for landing
        landed = False
        color_match = False
        
        # Collision with stack
        top_block = self.stacked_blocks[-1] if self.stacked_blocks else None
        if top_block:
            support_y = top_block["pos"][1] - top_block["size"][1] / 2
            if self.falling_block["pos"][1] + self.falling_block["size"][1] / 2 > support_y:
                landed = True
                if top_block["color"] == self.falling_block["color"]:
                    color_match = True
        # Collision with platform
        elif self.falling_block["pos"][1] + self.falling_block["size"][1] / 2 > self.PLATFORM_Y:
            landed = True

        if landed:
            self._handle_landing(color_match)
            return True, color_match
            
        return False, False

    def _handle_landing(self, color_match):
        new_block_data = self.falling_block
        self.falling_block = None

        # Create particles on impact
        self._create_particles(new_block_data["pos"], new_block_data["color"])

        # Finalize position and add to stack
        top_y = self.PLATFORM_Y
        top_block = self.stacked_blocks[-1] if self.stacked_blocks else None
        if top_block:
            top_y = top_block["pos"][1] - top_block["size"][1] / 2
        
        new_block_data["pos"][1] = top_y - new_block_data["size"][1] / 2
        new_block_data["angle"] = 0.0
        new_block_data["angular_vel"] = 0.0
        new_block_data["stability_timer"] = 0
        
        if color_match and top_block:
            new_block_data["stability_timer"] = self.STABILITY_TIMER_FRAMES
            top_block["stability_timer"] = self.STABILITY_TIMER_FRAMES
            
        self.stacked_blocks.append(new_block_data)
        self.momentum += self.MOMENTUM_PER_BLOCK
        
        if not self.game_over:
            self._spawn_falling_block()

    def _update_stack_stability(self):
        # Update stability timers
        for block in self.stacked_blocks:
            if block["stability_timer"] > 0:
                block["stability_timer"] -= 1

        # Simplified physics: check for toppling
        cumulative_mass_pos = np.array([0.0, 0.0])
        cumulative_mass = 0.0
        
        for i in range(len(self.stacked_blocks) - 1, -1, -1):
            block = self.stacked_blocks[i]
            mass = block["size"][0] * block["size"][1]
            
            cumulative_mass_pos += np.array(block["pos"]) * mass
            cumulative_mass += mass
            
            if cumulative_mass > 0:
                com_pos = cumulative_mass_pos / cumulative_mass
            else:
                continue

            # Support surface
            support_block = self.stacked_blocks[i-1] if i > 0 else None
            if support_block:
                support_x_min = support_block["pos"][0] - support_block["size"][0] / 2
                support_x_max = support_block["pos"][0] + support_block["size"][0] / 2
            else: # Platform
                support_x_min = self.SCREEN_WIDTH / 2 - self.PLATFORM_WIDTH / 2
                support_x_max = self.SCREEN_WIDTH / 2 + self.PLATFORM_WIDTH / 2

            # Check if COM is off the support
            if not (support_x_min < com_pos[0] < support_x_max):
                # Apply torque to all blocks above the support
                direction = 1 if com_pos[0] > (support_x_min + support_x_max) / 2 else -1
                for j in range(i, len(self.stacked_blocks)):
                    unstable_block = self.stacked_blocks[j]
                    if unstable_block["stability_timer"] == 0:
                        unstable_block["angular_vel"] += self.TOPPLE_FORCE * direction

        # Apply angular velocity and check for collapse
        for block in self.stacked_blocks:
            if block["stability_timer"] == 0:
                block["angle"] += math.degrees(block["angular_vel"])
                block["angular_vel"] *= self.ANGULAR_DAMPING

            if abs(block["angle"]) > self.COLLAPSE_ANGLE_THRESHOLD:
                return True # Collapse by angle
            
            # Check if any corner touches the ground
            corners = self._get_rotated_corners(block)
            for corner in corners:
                if corner[1] > self.GROUND_Y:
                    return True # Collapse by ground touch
        
        return False
        
    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed - 2]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "radius": self.np_random.uniform(2, 5),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += self.GRAVITY * 0.5 # Particles are lighter
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        # --- Background ---
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Platform ---
        base_rect = pygame.Rect(0, self.PLATFORM_Y + 5, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM_BASE, base_rect)
        
        tilt_rad = math.radians(self.platform_tilt)
        cx, cy = self.SCREEN_WIDTH / 2, self.PLATFORM_Y
        hw = self.PLATFORM_WIDTH / 2
        p1 = (cx - hw * math.cos(tilt_rad), cy - hw * math.sin(tilt_rad))
        p2 = (cx + hw * math.cos(tilt_rad), cy + hw * math.sin(tilt_rad))
        pygame.draw.line(self.screen, self.COLOR_PLATFORM_TOP, p1, p2, 6)

        # --- Stacked Blocks ---
        for block in self.stacked_blocks:
            self._draw_fancy_block(block)

        # --- Falling Block ---
        if self.falling_block:
            self._draw_fancy_block(self.falling_block)

        # --- Particles ---
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / 30.0))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])))

    def _draw_fancy_block(self, block):
        corners = self._get_rotated_corners(block)
        outline_color = tuple(c * self.BLOCK_OUTLINE_DARKEN for c in block["color"])
        
        # Glow for stabilized blocks
        if block.get("stability_timer", 0) > 0:
            glow_corners = self._get_rotated_corners(block, scale=1.2)
            glow_color = (*block["color"], 100)
            pygame.gfxdraw.filled_polygon(self.screen, glow_corners, glow_color)
        
        pygame.gfxdraw.filled_polygon(self.screen, corners, block["color"])
        pygame.gfxdraw.aapolygon(self.screen, corners, outline_color)

    def _get_rotated_corners(self, block, scale=1.0):
        w, h = block["size"]
        w *= scale
        h *= scale
        angle_rad = math.radians(block["angle"])
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        cx, cy = block["pos"]
        
        hw, hh = w / 2, h / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        
        rotated_corners = []
        for x, y in corners:
            rx = x * cos_a - y * sin_a + cx
            ry = x * sin_a + y * cos_a + cy
            rotated_corners.append((int(rx), int(ry)))
        return rotated_corners

    def _render_ui(self):
        # Momentum
        momentum_text = f"MOMENTUM: {int(self.momentum / self.MOMENTUM_PER_BLOCK * 1)}%"
        self._draw_text(momentum_text, (10, 10), self.font_small, self.COLOR_UI_TEXT)

        # Stack Height
        height_text = f"HEIGHT: {len(self.stacked_blocks)} / {self.WIN_STACK_HEIGHT}"
        text_surf = self.font_small.render(height_text, True, self.COLOR_UI_TEXT)
        self._draw_text(height_text, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 10), self.font_small, self.COLOR_UI_TEXT)
        
        # Game Over Text
        if self.game_over:
            if len(self.stacked_blocks) >= self.WIN_STACK_HEIGHT:
                msg = "YOU WIN!"
            else:
                msg = "STACK COLLAPSED"
            
            self._draw_text(msg, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 30), self.font_large, self.COLOR_UI_TEXT, center=True)

    def _draw_text(self, text, pos, font, color, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stack_height": len(self.stacked_blocks),
            "momentum": self.momentum,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # --- Manual Play ---
    # Re-enable the normal video driver for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Stacker")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
        
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.TARGET_FPS)

    env.close()