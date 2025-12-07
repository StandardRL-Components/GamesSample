
# Generated: 2025-08-27T16:47:34.452202
# Source Brief: brief_01330.md
# Brief Index: 1330

        
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
    """
    A procedurally generated side-view arcade game where the player controls a
    hopping space creature to reach the top platform.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use ← and → for a diagonal jump, or ↑ to jump straight up. "
        "Hold Shift for a power jump. Land on platforms to climb higher!"
    )

    # Short, user-facing description of the game
    game_description = (
        "A fast-paced arcade platformer. Hop your way up a tower of procedurally "
        "generated platforms before time runs out!"
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1200  # 40 seconds
        self.TIME_LIMIT_SECONDS = 30
        self.WORLD_WIDTH_FACTOR = 2.5 # How many screens wide the world is

        # Physics
        self.GRAVITY = 0.4
        self.FRICTION = -0.1
        self.JUMP_VELOCITY = -8
        self.POWER_JUMP_MODIFIER = 1.5
        self.JUMP_HORIZONTAL_VELOCITY = 4
        self.AIR_CONTROL_FORCE = 0.3
        self.MAX_VX = 6

        # Colors
        self.COLOR_BG_TOP = (10, 0, 30)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (50, 255, 50, 50)
        self.COLOR_PLATFORM = (100, 100, 120)
        self.COLOR_PLATFORM_EDGE = (100, 200, 255)
        self.COLOR_GOAL_PLATFORM = (255, 215, 0)
        self.COLOR_GOAL_PLATFORM_EDGE = (255, 255, 150)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (220, 220, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
        except pygame.error:
            self.font_large = pygame.font.SysFont("monospace", 24)
            self.font_small = pygame.font.SysFont("monospace", 16)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.jump_count = 0
        self.difficulty_level = 1.0

        self.player = {}
        self.platforms = []
        self.particles = []
        self.camera_offset_x = 0
        self.highest_platform_index = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT_SECONDS * self.FPS
        self.jump_count = 0

        # Reset game world
        self._generate_platforms()
        self.highest_platform_index = 0
        start_platform = self.platforms[0]
        self.player = {
            "x": start_platform.centerx,
            "y": start_platform.top,
            "vx": 0,
            "vy": 0,
            "radius": 10,
            "on_ground": True
        }
        self.particles.clear()
        self.camera_offset_x = 0

        if options and "difficulty" in options:
            self.difficulty_level = max(1.0, options["difficulty"])

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]
        shift_held = action[2] == 1

        # --- Update Game Logic ---
        self.steps += 1
        self.time_remaining -= 1
        reward = 0

        # 1. Handle Player Input
        if self.player["on_ground"]:
            jump_power = self.JUMP_VELOCITY * (self.POWER_JUMP_MODIFIER if shift_held else 1.0)
            jumped = False
            if movement == 1: # Up
                self.player["vy"] = jump_power
                self.player["vx"] = 0
                jumped = True
            elif movement == 3: # Left
                self.player["vy"] = jump_power
                self.player["vx"] = -self.JUMP_HORIZONTAL_VELOCITY
                jumped = True
            elif movement == 4: # Right
                self.player["vy"] = jump_power
                self.player["vx"] = self.JUMP_HORIZONTAL_VELOCITY
                jumped = True

            if jumped:
                # sfx: jump
                self.player["on_ground"] = False
                self.jump_count += 1
        else: # Air control
            if movement == 3: # Left
                self.player["vx"] -= self.AIR_CONTROL_FORCE
            elif movement == 4: # Right
                self.player["vx"] += self.AIR_CONTROL_FORCE

        # 2. Physics Update
        # Apply gravity
        self.player["vy"] += self.GRAVITY
        # Apply friction to horizontal velocity
        self.player["vx"] += self.player["vx"] * self.FRICTION
        self.player["vx"] = np.clip(self.player["vx"], -self.MAX_VX, self.MAX_VX)

        # Update position
        self.player["x"] += self.player["vx"]
        self.player["y"] += self.player["vy"]

        # 3. Collision Detection
        # World boundaries (horizontal)
        world_width = self.WIDTH * self.WORLD_WIDTH_FACTOR
        if self.player["x"] - self.player["radius"] < 0:
            self.player["x"] = self.player["radius"]
            self.player["vx"] *= -0.5 # Bounce
        elif self.player["x"] + self.player["radius"] > world_width:
            self.player["x"] = world_width - self.player["radius"]
            self.player["vx"] *= -0.5 # Bounce

        # Platform collisions
        self.player["on_ground"] = False
        player_rect = pygame.Rect(
            self.player["x"] - self.player["radius"],
            self.player["y"] - self.player["radius"],
            self.player["radius"] * 2,
            self.player["radius"] * 2
        )

        if self.player["vy"] > 0: # Only check for landing if falling
            for i, plat in enumerate(self.platforms):
                # Check if player's bottom is intersecting the platform's top surface
                if (player_rect.bottom > plat.top and
                    player_rect.bottom < plat.bottom and
                    player_rect.right > plat.left and
                    player_rect.left < plat.right):

                    self.player["y"] = plat.top - self.player["radius"]
                    self.player["vy"] = 0
                    self.player["on_ground"] = True
                    # sfx: land
                    self._create_particles(self.player["x"], plat.top, 10)

                    # Reward for landing on a higher platform
                    if i > self.highest_platform_index:
                        reward += 1.0
                        self.score += 100 * (i - self.highest_platform_index)
                        self.highest_platform_index = i

                    # Check for win condition
                    if i == len(self.platforms) - 1:
                        reward += 10.0
                        self.score += 5000
                        self.game_over = True
                    break

        # 4. Update Particles & Camera
        self._update_particles()
        self._update_camera()

        # 5. Continuous Reward
        if self.player["vy"] < 0:
            reward += 0.1 # Reward for moving up
        elif self.player["vy"] > 0:
            reward -= 0.01 # Small penalty for moving down

        # 6. Termination Check
        terminated = self.game_over
        if self.player["y"] - self.player["radius"] > self.HEIGHT:
            reward = -1.0
            terminated = True
        elif self.time_remaining <= 0:
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.score = max(0, self.score + int(reward))

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_platforms(self):
        self.platforms.clear()
        world_width = self.WIDTH * self.WORLD_WIDTH_FACTOR
        
        # Start platform
        start_w = 120
        start_x = (world_width - start_w) / 2
        start_y = self.HEIGHT - 40
        self.platforms.append(pygame.Rect(start_x, start_y, start_w, 20))

        # Procedural platforms
        num_platforms = 15
        min_y_gap = 60
        max_y_gap = 120
        max_x_offset = 200

        for i in range(1, num_platforms):
            last_plat = self.platforms[-1]
            
            # Increase difficulty
            current_min_y = min_y_gap + (i * 2 * self.difficulty_level)
            current_max_y = max_y_gap + (i * 3 * self.difficulty_level)
            
            new_y = last_plat.y - self.np_random.uniform(current_min_y, current_max_y)
            
            x_change = self.np_random.uniform(-max_x_offset, max_x_offset)
            new_x = last_plat.centerx + x_change - 50 # subtract half width
            
            new_w = self.np_random.uniform(70, 150)

            # Ensure platform is within world bounds
            new_x = np.clip(new_x, 50, world_width - new_w - 50)
            
            self.platforms.append(pygame.Rect(new_x, new_y, new_w, 20))
            
        # Goal platform (the last one)
        goal_plat = self.platforms[-1]
        goal_plat.width = 200
        goal_plat.x = (world_width - goal_plat.width) / 2
        goal_plat.y -= 20 # A bit higher

    def _create_particles(self, x, y, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                "x": x,
                "y": y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "lifespan": self.np_random.integers(15, 30),
                "radius": self.np_random.uniform(1, 3)
            })

    def _update_particles(self):
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += self.GRAVITY * 0.1 # Particles are lighter
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _update_camera(self):
        target_cam_x = self.player["x"] - self.WIDTH / 2
        world_width = self.WIDTH * self.WORLD_WIDTH_FACTOR
        
        # Clamp camera to world boundaries
        min_cam_x = 0
        max_cam_x = world_width - self.WIDTH
        clamped_cam_x = np.clip(target_cam_x, min_cam_x, max_cam_x)
        
        # Smooth camera movement
        self.camera_offset_x += (clamped_cam_x - self.camera_offset_x) * 0.1

    def _get_observation(self):
        # --- Render Background ---
        for y in range(self.HEIGHT):
            # Interpolate between top and bottom colors
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # --- Render Game Elements ---
        self._render_game()
        
        # --- Render UI ---
        self._render_ui()
        
        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_x = int(self.camera_offset_x)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = (*self.COLOR_PARTICLE, alpha)
            pos = (int(p["x"] - cam_x), int(p["y"]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"]), color)

        # Draw platforms
        for i, plat in enumerate(self.platforms):
            is_goal = i == len(self.platforms) - 1
            main_color = self.COLOR_GOAL_PLATFORM if is_goal else self.COLOR_PLATFORM
            edge_color = self.COLOR_GOAL_PLATFORM_EDGE if is_goal else self.COLOR_PLATFORM_EDGE
            
            # Draw main body
            rect_on_screen = plat.move(-cam_x, 0)
            pygame.draw.rect(self.screen, main_color, rect_on_screen)
            # Draw bright top edge
            pygame.draw.line(self.screen, edge_color,
                             (rect_on_screen.left, rect_on_screen.top),
                             (rect_on_screen.right, rect_on_screen.top), 2)

        # Draw player
        player_x_on_screen = int(self.player["x"] - cam_x)
        player_y_on_screen = int(self.player["y"])
        
        # Squash and stretch effect
        squash = 1.0 - min(0.4, max(-0.4, self.player["vy"] * 0.03))
        stretch = 1.0 + min(0.5, max(-0.4, self.player["vy"] * 0.03))
        radius_x = int(self.player["radius"] * stretch)
        radius_y = int(self.player["radius"] * squash)

        # Glow effect
        glow_radius = int(self.player["radius"] * 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (player_x_on_screen - glow_radius, player_y_on_screen - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Main player body
        player_rect = pygame.Rect(player_x_on_screen - radius_x, player_y_on_screen - radius_y, radius_x * 2, radius_y * 2)
        pygame.draw.ellipse(self.screen, self.COLOR_PLAYER, player_rect)
        
    def _render_ui(self):
        # Time remaining
        time_text = f"TIME: {self.time_remaining // self.FPS:02d}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Jumps made
        jumps_text = f"JUMPS: {self.jump_count}"
        jumps_surf = self.font_large.render(jumps_text, True, self.COLOR_TEXT)
        self.screen.blit(jumps_surf, (self.WIDTH - jumps_surf.get_width() - 10, 10))

        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "jumps": self.jump_count,
            "highest_platform": self.highest_platform_index,
            "difficulty": self.difficulty_level,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Hopper Game")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Print controls
    print("\n" + "="*30)
    print("      HUMAN PLAYING MODE")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Action mapping for human keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()

        clock.tick(env.FPS)

    print(f"Game Over! Final Info: {info}")
    env.close()
    pygame.quit()