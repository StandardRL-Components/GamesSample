import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:45:10.393429
# Source Brief: brief_00076.md
# Brief Index: 76
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a momentum-based platformer game.

    The player controls a character that must jump between platforms to collect
    10 checkpoints within a time limit. The core mechanic involves mastering
    momentum-based jumps, where collecting checkpoints increases jump power.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `actions[0]`: Aiming (0: none, 1: up, 2: down, 3: left, 4: right)
    - `actions[1]`: Short Jump (0: released, 1: pressed)
    - `actions[2]`: Long Jump (0: released, 1: pressed)

    **Observation Space:** `Box(0, 255, (400, 640, 3), dtype=uint8)`
    - A 640x400 RGB image of the game screen.

    **Rewards:**
    - +100 for winning (collecting 10 checkpoints).
    - -100 for losing (falling or running out of time).
    - +5 for using the one-time shortcut platform.
    - +1 for each checkpoint collected.
    - +0.1 for each successful landing on a platform.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A momentum-based platformer. Jump between platforms to collect checkpoints, which in turn increase your jump power."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to aim. Press space for a short jump and shift for a long jump."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000  # Approx. 66 seconds at 30 FPS, brief has conflicting values, using this for termination.
    TIME_LIMIT_SECONDS = 120

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (5, 10, 20)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 50)
    COLOR_PLATFORM = (100, 110, 130)
    COLOR_CHECKPOINT = (0, 255, 150)
    COLOR_CHECKPOINT_GLOW = (0, 255, 150, 80)
    COLOR_CHECKPOINT_COLLECTED = (60, 140, 120)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TRAJECTORY = (255, 255, 0, 100)

    # Physics
    GRAVITY = 0.5
    SHORT_JUMP_POWER = 9
    LONG_JUMP_POWER = 12
    AIM_SENSITIVITY = 0.08  # Radians per step

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        # Internal state variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.on_platform = None
        self.aim_angle = None
        self.momentum_multiplier = None
        self.checkpoints = None
        self.shortcut_triggered = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.platforms = []
        self._define_level()
        
        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging, not needed in final version

    def _define_level(self):
        """Defines the layout of platforms and checkpoints."""
        platform_data = [
            # x, y, width, has_checkpoint
            (50, 350, 100, False),  # Start
            (200, 320, 80, True),   # 1
            (350, 280, 80, True),   # 2
            (180, 220, 80, True),   # 3 (Shortcut Platform)
            (50, 150, 80, True),    # 4
            (250, 100, 80, True),   # 5
            (450, 150, 80, True),   # 6
            (530, 250, 80, True),   # 7 (Target of shortcut)
            (380, 340, 80, True),   # 8
            (500, 60, 60, True),    # 9
            (50, 50, 60, True)      # 10 (Final)
        ]
        
        self.platforms = []
        checkpoint_id = 0
        for i, (x, y, w, has_cp) in enumerate(platform_data):
            platform_rect = pygame.Rect(x, y, w, 15)
            is_shortcut = (i == 3) # Platform 3 is the shortcut
            is_shortcut_target = (i == 7) # Platform 7 is the target
            
            self.platforms.append({
                "rect": platform_rect,
                "has_checkpoint": has_cp,
                "checkpoint_id": checkpoint_id if has_cp else -1,
                "is_shortcut": is_shortcut,
                "is_shortcut_target": is_shortcut_target
            })
            if has_cp:
                checkpoint_id += 1
        
        self.total_checkpoints = checkpoint_id


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        start_platform_rect = self.platforms[0]['rect']
        self.player_pos = pygame.math.Vector2(start_platform_rect.centerx, start_platform_rect.top - 10)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.on_platform = self.platforms[0]
        self.aim_angle = -math.pi / 4  # Start aiming up-right
        self.momentum_multiplier = 1.0
        
        self.checkpoints = {p['checkpoint_id']: False for p in self.platforms if p['has_checkpoint']}
        self.shortcut_triggered = False

        self.steps = 0
        self.score = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0

        # --- 1. Handle Input ---
        self._handle_input(action)

        # --- 2. Update Physics ---
        landing_reward = self._update_physics()
        reward += landing_reward
        
        # --- 3. Check Interactions (Checkpoints, Shortcut) ---
        interaction_reward = self._check_interactions()
        reward += interaction_reward

        # --- 4. Check Termination Conditions ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.score >= self.total_checkpoints:
                reward = 100.0  # Win
            else:
                reward = -100.0 # Lose (fall or timeout)

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        """Process the MultiDiscrete action."""
        aim_action, short_jump_action, long_jump_action = action
        
        # Only allow aiming and jumping if on a platform
        if self.on_platform:
            # Aiming
            if aim_action == 1: # Up
                self.aim_angle = max(-math.pi + 0.1, self.aim_angle - self.AIM_SENSITIVITY)
            elif aim_action == 2: # Down
                self.aim_angle = min(-0.1, self.aim_angle + self.AIM_SENSITIVITY)
            elif aim_action == 3: # Left
                self.aim_angle -= self.AIM_SENSITIVITY
            elif aim_action == 4: # Right
                self.aim_angle += self.AIM_SENSITIVITY
            
            # Normalize angle to be within -pi to pi
            self.aim_angle = (self.aim_angle + math.pi) % (2 * math.pi) - math.pi
            
            # Jumping
            jump_triggered = short_jump_action == 1 or long_jump_action == 1
            if jump_triggered:
                # Prioritize long jump if both are pressed
                jump_power = self.LONG_JUMP_POWER if long_jump_action == 1 else self.SHORT_JUMP_POWER
                
                total_power = jump_power * self.momentum_multiplier
                
                self.player_vel.x = math.cos(self.aim_angle) * total_power
                self.player_vel.y = math.sin(self.aim_angle) * total_power
                
                self.on_platform = None
                # SFX: Jump sfx

    def _update_physics(self):
        """Update player position, handle gravity and platform collisions."""
        if self.on_platform is None:
            # Apply gravity
            self.player_vel.y += self.GRAVITY
            # Update position
            self.player_pos += self.player_vel

            # Check for landing
            player_rect = pygame.Rect(self.player_pos.x - 5, self.player_pos.y - 10, 10, 10)
            
            for platform in self.platforms:
                # Condition: moving down, player bottom is intersecting platform top
                if self.player_vel.y > 0 and platform['rect'].colliderect(player_rect):
                    # More precise check: was player above the platform last frame?
                    prev_pos_y = self.player_pos.y - self.player_vel.y
                    if prev_pos_y + 5 <= platform['rect'].top:
                        self.player_pos.y = platform['rect'].top - 5 # Snap to top
                        self.player_vel = pygame.math.Vector2(0, 0)
                        self.on_platform = platform
                        # SFX: Land sfx
                        return 0.1 # Reward for successful landing
        return 0.0

    def _check_interactions(self):
        """Check for checkpoint collection and shortcut activation."""
        if self.on_platform:
            platform = self.on_platform
            reward = 0
            
            # Check for shortcut
            if platform['is_shortcut'] and not self.shortcut_triggered:
                self.shortcut_triggered = True
                target_platform = next(p for p in self.platforms if p['is_shortcut_target'])
                self.player_pos = pygame.math.Vector2(target_platform['rect'].centerx, target_platform['rect'].top - 10)
                self.on_platform = target_platform
                # SFX: Teleport sfx
                reward += 5.0

            # Check for checkpoint
            if platform['has_checkpoint'] and not self.checkpoints[platform['checkpoint_id']]:
                self.checkpoints[platform['checkpoint_id']] = True
                self.score += 1
                self.momentum_multiplier += 0.10
                # SFX: Checkpoint sfx
                reward += 1.0
            
            return reward
        return 0.0

    def _check_termination(self):
        """Check for win or loss conditions."""
        # Win condition
        if self.score >= self.total_checkpoints:
            return True
        # Loss condition: fell off screen
        if self.player_pos.y > self.HEIGHT + 20:
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "momentum": self.momentum_multiplier
        }

    def _get_observation(self):
        """Render the game state to an RGB array."""
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        """Draw a vertical gradient for the background."""
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        """Render all game elements."""
        # Render platforms and checkpoints
        for p in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p['rect'], border_radius=3)
            if p['has_checkpoint']:
                center = p['rect'].centerx, p['rect'].top - 10
                if self.checkpoints[p['checkpoint_id']]:
                    pygame.gfxdraw.filled_circle(self.screen, int(center[0]), int(center[1]), 7, self.COLOR_CHECKPOINT_COLLECTED)
                else:
                    pygame.gfxdraw.filled_circle(self.screen, int(center[0]), int(center[1]), 12, self.COLOR_CHECKPOINT_GLOW)
                    pygame.gfxdraw.filled_circle(self.screen, int(center[0]), int(center[1]), 8, self.COLOR_CHECKPOINT)

        # Render jump trajectory if on a platform
        if self.on_platform:
            self._render_trajectory_arc()

        # Render player
        player_x, player_y = int(self.player_pos.x), int(self.player_pos.y)
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, 12, self.COLOR_PLAYER_GLOW)
        player_rect = pygame.Rect(player_x - 5, player_y - 5, 10, 10)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
    def _render_trajectory_arc(self):
        """Calculate and draw the predicted jump path."""
        power = self.SHORT_JUMP_POWER * self.momentum_multiplier
        vel = pygame.math.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * power
        pos = self.player_pos.copy()
        
        points = []
        for _ in range(15): # Predict 15 steps into the future
            vel.y += self.GRAVITY
            pos += vel
            points.append((int(pos.x), int(pos.y)))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.COLOR_TRAJECTORY, False, points, 2)


    def _render_ui(self):
        """Render the UI text elements."""
        # Checkpoint counter
        cp_text = f"CHECKPOINTS: {self.score} / {self.total_checkpoints}"
        cp_surf = self.font_small.render(cp_text, True, self.COLOR_TEXT)
        self.screen.blit(cp_surf, (10, 10))

        # Timer
        time_elapsed = self.steps / self.FPS
        time_left = max(0, self.TIME_LIMIT_SECONDS - time_elapsed)
        timer_text = f"TIME: {time_left:.1f}"
        timer_surf = self.font_small.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.score >= self.total_checkpoints else "GAME OVER"
            msg_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # It requires a graphical display.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Momentum Platformer")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    truncated = False
    
    while running:
        # --- Human Input ---
        aim_action = 0 # none
        short_jump = 0
        long_jump = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            aim_action = 1
        elif keys[pygame.K_DOWN]:
            aim_action = 2
        elif keys[pygame.K_LEFT]:
            aim_action = 3
        elif keys[pygame.K_RIGHT]:
            aim_action = 4
            
        if keys[pygame.K_SPACE]:
            short_jump = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            long_jump = 1

        action = [aim_action, short_jump, long_jump]

        # --- Environment Step ---
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to Display ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Reset on termination ---
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Final Reward: {reward}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            terminated = False
            truncated = False

        clock.tick(GameEnv.FPS)

    env.close()