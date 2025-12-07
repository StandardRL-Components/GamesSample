import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:12:48.196114
# Source Brief: brief_00952.md
# Brief Index: 952
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

# --- Helper Classes ---

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, pos, vel, life, color, radius_start, radius_end):
        self.pos = list(pos)
        self.vel = list(vel)
        self.life = life
        self.max_life = life
        self.color = color
        self.radius_start = radius_start
        self.radius_end = radius_end

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            life_ratio = self.life / self.max_life
            current_radius = int(self.radius_start * life_ratio + self.radius_end * (1 - life_ratio))
            if current_radius > 0:
                pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), current_radius, self.color)


class ClockPart:
    """Represents a part to be assembled (gear or hand)."""
    def __init__(self, part_type, target_pos, color, np_random):
        self.type = part_type
        self.pos = [-100, -100]
        self.target_pos = target_pos
        self.color = color
        self.state = 'unspawned'  # unspawned, on_conveyor, held, placed
        self.np_random = np_random
        self.size = 18 if self.type == 'gear' else (4, 40)

    def draw(self, surface, font):
        if self.state == 'unspawned':
            return

        draw_pos = (int(self.pos[0]), int(self.pos[1]))

        if self.state == 'placed':
            color = (50, 150, 255) # Blue for placed
        else:
            color = self.color

        if self.type == 'gear':
            pygame.gfxdraw.filled_circle(surface, draw_pos[0], draw_pos[1], self.size, color)
            pygame.gfxdraw.aacircle(surface, draw_pos[0], draw_pos[1], self.size, (255, 255, 255))
            # Draw some teeth for gear effect
            for i in range(8):
                angle = i * (math.pi / 4)
                p1 = (draw_pos[0] + self.size * math.cos(angle), draw_pos[1] + self.size * math.sin(angle))
                p2 = (draw_pos[0] + (self.size + 4) * math.cos(angle), draw_pos[1] + (self.size + 4) * math.sin(angle))
                pygame.draw.line(surface, color, p1, p2, 3)
        elif self.type == 'hand':
            # Hands are placed at an angle
            angle_rad = math.atan2(self.target_pos[1] - 320, self.target_pos[0] - 200)
            rect = pygame.Rect(0, 0, self.size[1], self.size[0])
            rect.center = draw_pos
            rotated_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            rotated_surf.fill(color)
            rotated_surf = pygame.transform.rotate(rotated_surf, -math.degrees(angle_rad))
            new_rect = rotated_surf.get_rect(center=draw_pos)
            surface.blit(rotated_surf, new_rect.topleft)


class RoboticArm:
    """Represents a robotic arm."""
    def __init__(self, arm_id, base_pos, np_random):
        self.id = arm_id
        self.base_pos = base_pos
        self.np_random = np_random
        
        self.length = 100
        self.angle = 0.0
        self.target_angle = 0.0
        self.end_effector_pos = [0, 0]

        self.state = 'idle'  # idle, moving, holding, broken
        self.speed_mode = 'slow'  # slow, fast
        self.held_part = None
        self.action_target_pos = None
        self.action_cooldown = 0

        self._update_end_effector()

    def _update_end_effector(self):
        self.end_effector_pos[0] = self.base_pos[0] + self.length * math.cos(self.angle)
        self.end_effector_pos[1] = self.base_pos[1] + self.length * math.sin(self.angle)

    def set_target_pos(self, pos):
        dx = pos[0] - self.base_pos[0]
        dy = pos[1] - self.base_pos[1]
        self.target_angle = math.atan2(dy, dx)
        self.action_target_pos = pos

    def update(self, particles):
        self.action_cooldown = max(0, self.action_cooldown - 1)
        
        if self.state == 'broken':
            return 0.0 # No reward change

        # Smoothly move towards target angle
        turn_speed = 0.05 if self.speed_mode == 'slow' else 0.12
        # Handle angle wrapping
        angle_diff = (self.target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        if abs(angle_diff) < turn_speed:
            self.angle = self.target_angle
        else:
            self.angle += np.sign(angle_diff) * turn_speed
        
        self._update_end_effector()

        # Visual effect for fast movement
        if self.state == 'moving' and self.speed_mode == 'fast':
            if self.np_random.random() < 0.3:
                p_vel = (self.np_random.random() - 0.5, self.np_random.random() - 0.5)
                particles.append(Particle(self.end_effector_pos, p_vel, 10, (255, 200, 0), 2, 0))

        # Malfunction check
        if self.state == 'moving' and self.speed_mode == 'fast':
            if self.np_random.random() < 0.002: # 0.2% chance per frame
                self.state = 'broken'
                # Malfunction sound placeholder: # sfx_arm_break()
                for _ in range(50):
                    angle = self.np_random.random() * 2 * math.pi
                    speed = self.np_random.random() * 5 + 2
                    vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                    life = self.np_random.integers(15, 30)
                    color = random.choice([(255, 50, 50), (255, 255, 255), (255, 150, 0)])
                    particles.append(Particle(self.end_effector_pos, vel, life, color, 4, 0))
                return -100 # Terminal penalty for breaking

        # Handle part interaction
        reward = 0.0
        if self.state == 'moving' and self.action_target_pos:
            dist_to_target = math.hypot(self.end_effector_pos[0] - self.action_target_pos[0], self.end_effector_pos[1] - self.action_target_pos[1])
            if dist_to_target < 10:
                if self.held_part: # Placing a part
                    # sfx_part_place()
                    self.held_part.pos = self.held_part.target_pos
                    self.held_part.state = 'placed'
                    reward += 1.0 if self.held_part.type == 'gear' else 2.0
                    self.held_part = None
                    self.state = 'idle'
                else: # Picking up a part
                    # This logic is handled in the main env loop to find the part
                    pass
        
        if self.held_part:
            self.held_part.pos = self.end_effector_pos

        return reward

    def draw(self, surface, is_selected):
        # Color based on state
        if self.state == 'broken':
            body_color = (139, 0, 0)
        elif self.speed_mode == 'fast':
            body_color = (255, 255, 0)
        else:
            body_color = (0, 200, 0)
        
        # Draw arm body
        pygame.draw.line(surface, body_color, self.base_pos, self.end_effector_pos, 8)
        
        # Draw base
        base_color = (100, 100, 100)
        if is_selected:
            pygame.gfxdraw.filled_circle(surface, int(self.base_pos[0]), int(self.base_pos[1]), 20, (255, 255, 255))
            pygame.gfxdraw.aacircle(surface, int(self.base_pos[0]), int(self.base_pos[1]), 20, (255, 255, 255))
        pygame.gfxdraw.filled_circle(surface, int(self.base_pos[0]), int(self.base_pos[1]), 18, base_color)
        pygame.gfxdraw.filled_circle(surface, int(self.base_pos[0]), int(self.base_pos[1]), 14, body_color)

        # Draw gripper
        gripper_size = 10
        angle_perp = self.angle + math.pi / 2
        p1 = (self.end_effector_pos[0] + gripper_size * math.cos(angle_perp), self.end_effector_pos[1] + gripper_size * math.sin(angle_perp))
        p2 = (self.end_effector_pos[0] - gripper_size * math.cos(angle_perp), self.end_effector_pos[1] - gripper_size * math.sin(angle_perp))
        pygame.draw.line(surface, (200, 200, 200), p1, p2, 6)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Assemble a clock against the timer by controlling robotic arms to pick up and place gears and hands from a conveyor belt."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select an arm. Press space to toggle speed. Hold shift to pick up or place a part."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME_SECONDS = 90
        self.MAX_STEPS = self.FPS * self.MAX_TIME_SECONDS # Adjusted max steps to match timer

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_CONVEYOR = (60, 70, 80)
        self.COLOR_CLOCK_FRAME = (100, 110, 120)
        self.COLOR_TEXT = (220, 220, 220)
        
        # State variables are initialized in reset()
        self.arms = []
        self.parts = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.selected_arm_idx = 0
        self.last_space_held = False
        self.conveyor_pos = 0

        self.reset()
        # self.validate_implementation() # Commented out for submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_TIME_SECONDS * self.FPS
        self.selected_arm_idx = 0
        self.last_space_held = False
        self.conveyor_pos = 0
        
        self.particles.clear()
        self.arms.clear()
        self.parts.clear()

        # Initialize arms
        arm_bases = [(self.WIDTH // 2, 50), (self.WIDTH // 2, self.HEIGHT - 50), (50, self.HEIGHT // 2), (self.WIDTH - 50, self.HEIGHT // 2)]
        for i in range(4):
            self.arms.append(RoboticArm(i, arm_bases[i], self.np_random))

        # Initialize parts
        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
        part_defs = [
            ('gear', (center_x, center_y - 60), (200, 150, 50)),
            ('gear', (center_x, center_y + 60), (200, 150, 50)),
            ('gear', (center_x - 60, center_y), (200, 150, 50)),
            ('gear', (center_x + 60, center_y), (200, 150, 50)),
            ('hand', (center_x + 70, center_y), (200, 50, 50)), # Hour hand
            ('hand', (center_x, center_y - 80), (200, 50, 50)), # Minute hand
        ]
        for i, (part_type, target_pos, color) in enumerate(part_defs):
            part = ClockPart(part_type, target_pos, color, self.np_random)
            # Place on conveyor belt
            part.pos = [self.WIDTH + 50 + i * 100, self.HEIGHT - 25]
            part.state = 'on_conveyor'
            self.parts.append(part)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement in [1, 2, 3, 4]:
            self.selected_arm_idx = movement - 1

        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        selected_arm = self.arms[self.selected_arm_idx]

        if space_pressed and selected_arm.state != 'broken':
            # sfx_toggle_speed()
            selected_arm.speed_mode = 'fast' if selected_arm.speed_mode == 'slow' else 'slow'

        if shift_held and selected_arm.state not in ['moving', 'broken'] and selected_arm.action_cooldown == 0:
            # sfx_arm_activate()
            if selected_arm.held_part is None:
                # Find closest available part
                available_parts = [p for p in self.parts if p.state == 'on_conveyor' and 0 < p.pos[0] < self.WIDTH]
                if available_parts:
                    closest_part = min(available_parts, key=lambda p: math.hypot(p.pos[0] - selected_arm.base_pos[0], p.pos[1] - selected_arm.base_pos[1]))
                    dist_to_part = math.hypot(closest_part.pos[0] - selected_arm.base_pos[0], closest_part.pos[1] - selected_arm.base_pos[1])
                    if dist_to_part <= selected_arm.length + closest_part.size:
                        selected_arm.set_target_pos(closest_part.pos)
                        selected_arm.state = 'moving'
                        selected_arm.action_cooldown = 15 # 0.5s cooldown
            else: # If holding a part
                selected_arm.set_target_pos(selected_arm.held_part.target_pos)
                selected_arm.state = 'moving'
                selected_arm.action_cooldown = 15

        # --- Update Game State ---
        self.time_remaining -= 1
        
        # Update conveyor
        self.conveyor_pos = (self.conveyor_pos + 1) % 40
        for part in self.parts:
            if part.state == 'on_conveyor':
                part.pos[0] -= 1

        # Update arms and check for interactions
        for arm in self.arms:
            arm_reward = arm.update(self.particles)
            if arm_reward < 0: # Arm broke
                self.game_over = True
                reward = arm_reward
                break
            reward += arm_reward

            # Check for pickup
            if arm.state == 'moving' and arm.held_part is None and arm.action_target_pos:
                for part in self.parts:
                    if part.state == 'on_conveyor' and part.pos == arm.action_target_pos:
                        dist_to_part = math.hypot(arm.end_effector_pos[0] - part.pos[0], arm.end_effector_pos[1] - part.pos[1])
                        if dist_to_part < 15:
                            # sfx_part_pickup()
                            arm.held_part = part
                            part.state = 'held'
                            arm.state = 'holding'
                            # Immediately set new target to dropoff
                            arm.set_target_pos(part.target_pos)
                            arm.state = 'moving'
                            break
        if self.game_over: # Early exit if an arm broke
            self.score += reward
            return self._get_observation(), reward, True, False, self._get_info()

        # Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

        self.score += reward
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if all(p.state == 'placed' for p in self.parts):
                # sfx_win()
                reward += 100
            else: # Timeout
                # sfx_lose()
                reward -= 100
            self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if any(arm.state == 'broken' for arm in self.arms):
            return True
        if self.time_remaining <= 0:
            return True
        if all(p.state == 'placed' for p in self.parts):
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left_seconds": self.time_remaining // self.FPS}

    def _render_game(self):
        # Background grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Conveyor belt
        conveyor_rect = pygame.Rect(0, self.HEIGHT - 50, self.WIDTH, 50)
        pygame.draw.rect(self.screen, self.COLOR_CONVEYOR, conveyor_rect)
        for i in range(-1, self.WIDTH // 40 + 1):
            x = i * 40 + self.conveyor_pos
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.HEIGHT - 50), (x, self.HEIGHT), 2)

        # Clock frame
        center = (self.WIDTH // 2, self.HEIGHT // 2)
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], 100, self.COLOR_CLOCK_FRAME)
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], 98, self.COLOR_CLOCK_FRAME)

        # Parts
        for part in self.parts:
            part.draw(self.screen, self.font_small)

        # Arms
        for i, arm in enumerate(self.arms):
            arm.draw(self.screen, i == self.selected_arm_idx)

        # Particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_sec = self.time_remaining // self.FPS
        time_msec = (self.time_remaining % self.FPS) * 100 // self.FPS
        timer_color = (255, 100, 100) if time_sec < 10 else self.COLOR_TEXT
        timer_text = self.font_large.render(f"{time_sec:02}:{time_msec:02}", True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Selected Arm Info
        arm = self.arms[self.selected_arm_idx]
        status_text = f"ARM {arm.id + 1} | SPEED: {arm.speed_mode.upper()} | STATUS: {arm.state.upper()}"
        status_render = self.font_small.render(status_text, True, self.COLOR_TEXT)
        self.screen.blit(status_render, (10, self.HEIGHT - status_render.get_height() - 60))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = all(p.state == 'placed' for p in self.parts)
            msg = "ASSEMBLY COMPLETE" if win_condition else "FAILURE"
            color = (100, 255, 100) if win_condition else (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def render(self):
        return self._get_observation()
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you need to unset the dummy video driver
    # and install pygame with display support.
    # For example:
    # pip install pygame
    # unset SDL_VIDEODRIVER
    
    # We re-set the environment variable to run the interactive mode
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Clockwork Assembly")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Action state
    movement = 0 # 0: none, 1: up, 2: down, 3: left, 4: right
    space_held = 0 # 0: released, 1: held
    shift_held = 0 # 0: released, 1: held

    print("\n--- Manual Control ---")
    print("Arrow Keys: Select Arm")
    print("Spacebar: Toggle Speed")
    print("Shift: Perform Action")
    print("R: Reset Environment")
    print("Q: Quit")
    print("----------------------")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                
                # Update actions on key down
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                if event.key == pygame.K_SPACE: space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1

            if event.type == pygame.KEYUP:
                # Reset non-movement actions on key up
                if event.key == pygame.K_SPACE: space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 0
                # Reset movement to none if the corresponding key is released
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    movement = 0

        if not terminated:
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # For manual play, we want to reset movement after one step
            movement = 0

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()