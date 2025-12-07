import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:14:09.030383
# Source Brief: brief_03389.md
# Brief Index: 3389
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a tower-building puzzle game.

    The goal is to build the tallest tower possible by stacking falling blocks.
    Strategic placement is key, as stacking three identical blocks vertically
    creates a stronger, synchronized block, awarding bonus points and improving
    stability. The tower sways based on block placement, and can collapse if
    it becomes too unstable.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0=none, 1=up(unused), 2=down(unused), 3=left, 4=right)
    - action[1]: Place Block (0=released, 1=pressed)
    - action[2]: Shift (unused)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Rewards:
    - +0.1 for successfully placing a block.
    - +1.0 for creating a synchronized (3-stack) block.
    - -5.0 for placing a block that significantly increases instability.
    - +100.0 for winning by reaching the target height.
    - -50.0 if the tower collapses.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack falling blocks to build the tallest tower possible. Create stable 3-block stacks of the same color for bonus points, but be careful of the tower swaying and collapsing."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to move the falling block. Press space to drop the block instantly."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_HEIGHT = 100
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (26, 26, 46)
    COLOR_GRID = (42, 42, 78)
    COLOR_UI_TEXT = (230, 230, 240)
    COLOR_TARGET_LINE = (255, 200, 0)
    BLOCK_COLORS = [
        (255, 70, 70),   # Red
        (70, 255, 70),   # Green
        (70, 70, 255),   # Blue
        (255, 255, 70),  # Yellow
    ]
    SYNC_COLORS = [
        (255, 150, 150),
        (150, 255, 150),
        (150, 150, 255),
        (255, 255, 150),
    ]

    # Game parameters
    BLOCK_WIDTH = 40
    BLOCK_HEIGHT = 15
    PLAYER_SPEED = 8
    INITIAL_FALL_SPEED = 1.0 # Units per step
    FALL_SPEED_INCREASE = 0.05 / 500 # per step
    SWAY_TORQUE_FACTOR = 0.05
    SWAY_DAMPING = 0.99
    SWAY_SPEED = 0.05
    MAX_SWAY_MAGNITUDE = BLOCK_WIDTH * 0.75
    INSTABILITY_THRESHOLD = BLOCK_WIDTH * 0.4
    PARTICLE_LIFESPAN = 30
    PARTICLE_COUNT = 20

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        
        # State variables are initialized in reset()
        self.placed_blocks = []
        self.falling_block = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tower_height = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.camera_y = 0
        self.tower_sway_magnitude = 0.0
        self.tower_sway_phase = 0.0
        self.particles = []
        self.prev_space_held = False
        self.last_reward_info = ""

        # This call is not strictly necessary but good practice
        self.reset()
        
        # Critical self-check
        # self.validate_implementation() # Commented out for submission


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.camera_y = 0
        self.tower_sway_magnitude = 0.0
        self.tower_sway_phase = 0.0
        self.particles = []
        self.prev_space_held = False
        self.last_reward_info = ""

        # Create the ground platform
        ground_rect = pygame.Rect(0, self.SCREEN_HEIGHT - self.BLOCK_HEIGHT, self.SCREEN_WIDTH, self.BLOCK_HEIGHT)
        self.placed_blocks = [{
            "rect": ground_rect,
            "color_id": -1, # Special ID for ground
            "synced": True,
            "level": 0
        }]
        self.tower_height = 0

        self._generate_new_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False
        self.steps += 1
        self.last_reward_info = ""

        # --- Update Game State ---
        self._handle_input(action)
        self._update_falling_block()
        self._update_sway()
        self._update_particles()
        
        # --- Check for placement ---
        placement_info = self._check_placement()
        if placement_info["placed"]:
            placement_reward = self._handle_placement(placement_info)
            reward += placement_reward
            self._generate_new_block()

        # --- Check for Termination ---
        if self.tower_height >= self.TARGET_HEIGHT:
            reward += 100.0
            self.last_reward_info = "WIN! (+100)"
            terminated = True
        elif self._check_collapse():
            reward -= 50.0
            self.last_reward_info = "COLLAPSE! (-50)"
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        
        # --- Store action state for next step ---
        self.prev_space_held = (action[1] == 1)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_pressed = (action[1] == 1) and not self.prev_space_held

        # Horizontal movement
        if movement == 3: # Left
            self.falling_block["rect"].x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.falling_block["rect"].x += self.PLAYER_SPEED

        # Clamp falling block to screen bounds
        self.falling_block["rect"].x = max(0, min(self.SCREEN_WIDTH - self.BLOCK_WIDTH, self.falling_block["rect"].x))
        
        # Vertical movement (fast drop)
        if space_pressed:
            # Drop the block instantly onto the surface below
            _, landing_y, _ = self._get_landing_site()
            self.falling_block["rect"].y = landing_y - self.BLOCK_HEIGHT
            # Sound: Fast drop sfx

    def _update_falling_block(self):
        # Increase fall speed over time
        self.fall_speed += self.FALL_SPEED_INCREASE
        self.falling_block["rect"].y += self.fall_speed

    def _update_sway(self):
        self.tower_sway_phase += self.SWAY_SPEED
        self.tower_sway_magnitude *= self.SWAY_DAMPING

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1

    def _check_placement(self):
        landing_site, landing_y, support_block_idx = self._get_landing_site()
        
        if self.falling_block["rect"].bottom >= landing_y:
            return {
                "placed": True,
                "final_y": landing_y - self.BLOCK_HEIGHT,
                "support_block_idx": support_block_idx
            }
        return {"placed": False}

    def _get_landing_site(self):
        highest_y = self.SCREEN_HEIGHT
        support_block_idx = -1 # -1 indicates ground
        
        # Find the highest point on the tower underneath the falling block
        for i, block in enumerate(self.placed_blocks):
            if self.falling_block["rect"].colliderect(block["rect"].x, block["rect"].y - self.SCREEN_HEIGHT, block["rect"].width, self.SCREEN_HEIGHT):
                if block["rect"].top < highest_y:
                    highest_y = block["rect"].top
                    support_block_idx = i
        return self.placed_blocks[support_block_idx], highest_y, support_block_idx

    def _handle_placement(self, placement_info):
        reward = 0.1 # Base reward for any placement
        self.last_reward_info = "Place (+0.1)"
        
        self.falling_block["rect"].y = placement_info["final_y"]
        
        support_block = self.placed_blocks[placement_info["support_block_idx"]]
        new_block_level = support_block["level"] + 1

        new_block = {
            "rect": self.falling_block["rect"].copy(),
            "color_id": self.falling_block["color_id"],
            "synced": False,
            "level": new_block_level
        }
        self.placed_blocks.append(new_block)
        # Sound: Block place sfx
        
        # Check for synchronization
        sync_reward = self._check_synchronization(len(self.placed_blocks) - 1)
        if sync_reward > 0:
            reward += sync_reward
            self.last_reward_info += f" Sync (+{sync_reward:.1f})"
        
        # Update tower stability
        stability_penalty = self._update_tower_stability(new_block, support_block)
        if stability_penalty < 0:
            reward += stability_penalty
            self.last_reward_info += f" Unstable ({stability_penalty:.1f})"
        
        # Update tower height and camera
        self.tower_height = max(self.tower_height, new_block_level)
        self.camera_y = max(0, (self.tower_height * self.BLOCK_HEIGHT) - (self.SCREEN_HEIGHT * 0.6))

        self.score += reward
        return reward

    def _check_synchronization(self, new_block_idx):
        block1 = self.placed_blocks[new_block_idx]
        if block1["level"] < 3:
            return 0

        # Find blocks 2 and 3 levels below at the same approximate x-position
        block2 = None
        block3 = None
        
        # This is O(N), but N is max ~100. For larger towers, a grid would be better.
        for block in self.placed_blocks:
            if block["level"] == block1["level"] - 1 and abs(block["rect"].centerx - block1["rect"].centerx) < self.BLOCK_WIDTH / 2:
                block2 = block
            elif block["level"] == block1["level"] - 2 and abs(block["rect"].centerx - block1["rect"].centerx) < self.BLOCK_WIDTH / 2:
                block3 = block
        
        if block2 and block3:
            if block1["color_id"] == block2["color_id"] == block3["color_id"]:
                if not block1["synced"] and not block2["synced"] and not block3["synced"]:
                    block1["synced"] = block2["synced"] = block3["synced"] = True
                    self._spawn_particles(block1["rect"].center, block1["color_id"])
                    # Sound: Sync success sfx
                    return 1.0
        return 0

    def _update_tower_stability(self, new_block, support_block):
        penalty = 0
        if support_block["color_id"] != -1: # Not the ground
            offset = new_block["rect"].centerx - support_block["rect"].centerx
            # Add "torque" to the sway. More torque for higher placements.
            self.tower_sway_magnitude += abs(offset) * self.SWAY_TORQUE_FACTOR * (1 + new_block["level"] / 50)
            
            # Penalize large increases in instability
            if self.tower_sway_magnitude > self.INSTABILITY_THRESHOLD:
                penalty = -5.0
                # Sound: Instability warning sfx
        
        self.tower_sway_magnitude = min(self.MAX_SWAY_MAGNITUDE, self.tower_sway_magnitude)
        return penalty

    def _check_collapse(self):
        # Simple collapse: a block is placed with no support
        last_block = self.placed_blocks[-1]
        support_block = self.placed_blocks[self._get_landing_site()[2]]
        
        if support_block['color_id'] == -1: # Ground
             return False

        sway_at_support = self._get_sway_offset(support_block['level'])
        sway_at_last = self._get_sway_offset(last_block['level'])

        support_left = support_block['rect'].left + sway_at_support
        support_right = support_block['rect'].right + sway_at_support
        
        last_left = last_block['rect'].left + sway_at_last
        last_right = last_block['rect'].right + sway_at_last

        # Check if the new block is supported by less than 20% of its width
        overlap = max(0, min(last_right, support_right) - max(last_left, support_left))
        if overlap < self.BLOCK_WIDTH * 0.2 and not support_block["synced"]:
            return True

        return False

    def _generate_new_block(self):
        self.falling_block = {
            "rect": pygame.Rect(self.SCREEN_WIDTH / 2 - self.BLOCK_WIDTH / 2, -self.BLOCK_HEIGHT - self.camera_y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
            "color_id": self.np_random.integers(0, len(self.BLOCK_COLORS))
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_background_grid()
        self._draw_target_line()
        
        # Current sway for this frame
        current_sway = math.sin(self.tower_sway_phase) * self.tower_sway_magnitude

        # Draw placed blocks
        for block in self.placed_blocks:
            if block["color_id"] == -1: # Ground
                pygame.draw.rect(self.screen, (50, 50, 80), block["rect"])
            else:
                sway_offset = self._get_sway_offset(block['level'])
                
                screen_y = block["rect"].y - self.camera_y
                
                # Culling
                if screen_y > self.SCREEN_HEIGHT or screen_y < -self.BLOCK_HEIGHT:
                    continue

                draw_rect = block["rect"].copy()
                draw_rect.x += sway_offset
                draw_rect.y = screen_y
                
                self._draw_block(self.screen, draw_rect, block["color_id"], block["synced"])
        
        # Draw falling block
        if self.falling_block:
            draw_rect = self.falling_block["rect"].copy()
            draw_rect.y -= self.camera_y
            self._draw_block(self.screen, draw_rect, self.falling_block["color_id"], False, is_falling=True)
            
            # Draw landing indicator
            _, landing_y, _ = self._get_landing_site()
            indicator_y = landing_y - self.camera_y
            indicator_rect = pygame.Rect(self.falling_block['rect'].x, indicator_y - 2, self.BLOCK_WIDTH, 2)
            pygame.draw.rect(self.screen, (255, 255, 255, 150), indicator_rect)

        # Draw particles
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1] - self.camera_y))
            alpha = int(255 * (p["lifespan"] / self.PARTICLE_LIFESPAN))
            color = self.BLOCK_COLORS[p["color_id"]]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, (*color, alpha))

    def _get_sway_offset(self, level):
        # No sway for ground
        if level <= 0:
            return 0
        # Linear interpolation of sway effect from base to top
        sway_effect = (level / self.tower_height) if self.tower_height > 0 else 0
        return math.sin(self.tower_sway_phase) * self.tower_sway_magnitude * sway_effect

    def _draw_background_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.BLOCK_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.BLOCK_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_target_line(self):
        target_y = self.SCREEN_HEIGHT - self.BLOCK_HEIGHT - (self.TARGET_HEIGHT * self.BLOCK_HEIGHT) - self.camera_y
        if 0 < target_y < self.SCREEN_HEIGHT:
            for x in range(0, self.SCREEN_WIDTH, 20):
                pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (x, target_y), (x + 10, target_y), 2)

    def _draw_block(self, surface, rect, color_id, synced, is_falling=False):
        main_color = self.SYNC_COLORS[color_id] if synced else self.BLOCK_COLORS[color_id]
        
        if synced:
            # Glow effect
            glow_rect = rect.inflate(8, 8)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            glow_color = (*main_color, 50)
            pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=5)
            surface.blit(glow_surf, glow_rect.topleft)

        # Main block body with border
        border_color = tuple(max(0, c-40) for c in main_color)
        pygame.draw.rect(surface, border_color, rect, border_radius=3)
        inner_rect = rect.inflate(-4, -4)
        pygame.draw.rect(surface, main_color, inner_rect, border_radius=3)

        if is_falling:
            # Add a highlight to the falling block
            highlight_color = (255, 255, 255, 100)
            pygame.draw.rect(surface, highlight_color, inner_rect, border_radius=3)

    def _render_ui(self):
        # Score display
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))

        # Height display
        height_text = f"HEIGHT: {self.tower_height}/{self.TARGET_HEIGHT}"
        height_surf = self.font_ui.render(height_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(height_surf, (10, 10))
        
        # Reward info for debugging
        # reward_surf = self.font_ui.render(self.last_reward_info, True, self.COLOR_UI_TEXT)
        # self.screen.blit(reward_surf, (10, 35))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.tower_height >= self.TARGET_HEIGHT else "GAME OVER"
            text_surf = self.font_game_over.render(message, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.tower_height,
            "fall_speed": self.fall_speed,
            "sway": self.tower_sway_magnitude
        }
        
    def _spawn_particles(self, pos, color_id):
        for _ in range(self.PARTICLE_COUNT):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": self.PARTICLE_LIFESPAN,
                "color_id": color_id
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == '__main__':
    # This block will not run in the test environment, but is useful for local testing.
    # To run, you'll need to `pip install pygame` and unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Controls: Left/Right Arrow Keys, Space to drop
    pygame.display.set_caption("Tower Builder")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        # Construct action from keyboard input
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Unused
        
        action = [movement, space_held, shift_held]

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- ENV RESET ---")

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # The game will show a "GAME OVER" screen, press R to reset
        
        # Render the observation from the environment to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()