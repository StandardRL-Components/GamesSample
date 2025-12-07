import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:19:33.687835
# Source Brief: brief_02693.md
# Brief Index: 2693
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    game_description = (
        "Control the magnetic polarity of two concentric gears to prevent them from colliding by manipulating attraction and repulsion forces."
    )
    user_guide = (
        "Press 'space' to cycle the large gear's magnetic state and 'shift' to cycle the small gear's state. Avoid collision!"
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 20
        self.MAX_STEPS = 180 * self.FPS  # 180 seconds

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_LARGE_GEAR = (50, 150, 255)
        self.COLOR_SMALL_GEAR = (255, 80, 80)
        self.COLOR_ATTRACT = (255, 220, 50)
        self.COLOR_REPEL = (50, 255, 150)
        self.COLOR_SPARK = (255, 255, 255)

        # Game Parameters
        self.CENTER = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.LARGE_GEAR_RADIUS = 150
        self.SMALL_GEAR_RADIUS = 100
        self.GEAR_NODE_RADIUS = 12
        self.GEAR_SPOKE_COUNT = 12
        
        # Physics
        self.MAGNETIC_FORCE_STRENGTH = 0.00015
        self.INERTIA_LARGE = 1.5
        self.INERTIA_SMALL = 1.0
        self.VELOCITY_DAMPING = 0.998
        self.COLLISION_ANGLE_THRESHOLD = math.radians(10) # Angle for collision

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        self.font_status = pygame.font.Font(None, 20)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.collision_occured = False
        self.large_gear_angle = 0.0
        self.large_gear_velocity = 0.0
        self.large_gear_magnetism = 0 # 0:off, 1:attract, 2:repel
        self.small_gear_angle = 0.0
        self.small_gear_velocity = 0.0
        self.small_gear_magnetism = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.sparks = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.collision_occured = False

        # Initial gear states
        self.large_gear_angle = 0.0
        self.large_gear_velocity = (10 / 60) * 2 * math.pi / self.FPS # 10 RPM
        self.large_gear_magnetism = 0

        self.small_gear_angle = math.pi # Start 180 degrees apart
        self.small_gear_velocity = (20 / 60) * 2 * math.pi / self.FPS # 20 RPM
        self.small_gear_magnetism = 0

        self.prev_space_held = False
        self.prev_shift_held = False
        self.sparks = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        # movement = action[0] # Unused
        space_held = action[1] == 1
        shift_held = action[2] == 1

        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        if space_pressed:
            # SFX: UI Click
            self.large_gear_magnetism = (self.large_gear_magnetism + 1) % 3
        if shift_pressed:
            # SFX: UI Click
            self.small_gear_magnetism = (self.small_gear_magnetism + 1) % 3
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Game Logic Update ---
        self._apply_physics()
        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        win = self.steps >= self.MAX_STEPS
        self.collision_occured = self._check_collision()

        if self.collision_occured or win:
            terminated = True
            self.game_over = True
            if self.collision_occured:
                # SFX: Explosion/Collision
                self._create_sparks()

        # --- Reward Calculation ---
        reward = self._calculate_reward(terminated, win)
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _apply_physics(self):
        # Calculate angular difference, handling wrap-around
        angle_diff = (self.small_gear_angle - self.large_gear_angle + math.pi) % (2 * math.pi) - math.pi

        # Determine interaction type
        interaction_type = 0 # 0: none, 1: attract, -1: repel
        if self.large_gear_magnetism > 0 and self.small_gear_magnetism > 0:
            if self.large_gear_magnetism == 1 and self.small_gear_magnetism == 1: # attract-attract
                interaction_type = 1
            else: # any other combo is repel
                interaction_type = -1
        
        # Calculate force
        # Attraction pulls nodes together, repulsion pushes them apart
        force = interaction_type * self.MAGNETIC_FORCE_STRENGTH * math.sin(angle_diff)

        # Apply force to velocities (dv = F/I)
        self.large_gear_velocity += force / self.INERTIA_LARGE
        self.small_gear_velocity -= force / self.INERTIA_SMALL

        # Apply damping
        self.large_gear_velocity *= self.VELOCITY_DAMPING
        self.small_gear_velocity *= self.VELOCITY_DAMPING

        # Update angles
        self.large_gear_angle = (self.large_gear_angle + self.large_gear_velocity) % (2 * math.pi)
        self.small_gear_angle = (self.small_gear_angle + self.small_gear_velocity) % (2 * math.pi)

    def _check_collision(self):
        angle_diff = abs((self.small_gear_angle - self.large_gear_angle + math.pi) % (2 * math.pi) - math.pi)
        return angle_diff < self.COLLISION_ANGLE_THRESHOLD

    def _calculate_reward(self, terminated, win):
        if not terminated:
            reward = 0.1  # Continuous reward for survival
            if self.steps > 0 and self.steps % self.FPS == 0:
                reward += 1.0 # Bonus for each second survived
            return reward
        else:
            if self.collision_occured:
                return -100.0 # Penalty for collision
            elif win:
                return 100.0 # Reward for winning
        return 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background grid
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # Draw center axis
        pygame.gfxdraw.filled_circle(self.screen, self.CENTER[0], self.CENTER[1], 5, self.COLOR_GRID)

        # Draw magnetic auras
        self._draw_aura(self.large_gear_magnetism, self.LARGE_GEAR_RADIUS, self.COLOR_ATTRACT, self.COLOR_REPEL)
        self._draw_aura(self.small_gear_magnetism, self.SMALL_GEAR_RADIUS, self.COLOR_ATTRACT, self.COLOR_REPEL)

        # Draw gears
        self._draw_gear(self.LARGE_GEAR_RADIUS, self.large_gear_angle, self.COLOR_LARGE_GEAR)
        self._draw_gear(self.SMALL_GEAR_RADIUS, self.small_gear_angle, self.COLOR_SMALL_GEAR)

        # Draw collision sparks
        if self.collision_occured:
            self._update_and_draw_sparks()

    def _draw_aura(self, magnetism_type, radius, attract_color, repel_color):
        if magnetism_type == 0:
            return
        
        color = attract_color if magnetism_type == 1 else repel_color
        # SFX: Hum/Buzz loop while active
        
        # Pulsating effect
        pulse_alpha = (math.sin(self.steps * 0.3) + 1) / 2 # Varies between 0 and 1
        alpha = 30 + pulse_alpha * 40 # Varies between 30 and 70

        # Create a temporary surface for the aura to handle transparency
        aura_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        
        # Draw multiple circles for a soft glow effect
        for i in range(4):
            current_radius = int(radius + i * 4 - 6)
            current_alpha = int(alpha * (1 - i*0.2))
            if current_radius > 0 and current_alpha > 0:
                 pygame.gfxdraw.aacircle(aura_surface, self.CENTER[0], self.CENTER[1], current_radius, (*color, current_alpha))

        self.screen.blit(aura_surface, (0, 0))

    def _draw_gear(self, radius, angle, color):
        # Draw path
        pygame.gfxdraw.aacircle(self.screen, self.CENTER[0], self.CENTER[1], radius, self.COLOR_GRID)

        # Draw spokes
        for i in range(self.GEAR_SPOKE_COUNT):
            spoke_angle = angle + (i * 2 * math.pi / self.GEAR_SPOKE_COUNT)
            start_x = self.CENTER[0] + 20 * math.cos(spoke_angle)
            start_y = self.CENTER[1] + 20 * math.sin(spoke_angle)
            end_x = self.CENTER[0] + (radius - self.GEAR_NODE_RADIUS / 2) * math.cos(spoke_angle)
            end_y = self.CENTER[1] + (radius - self.GEAR_NODE_RADIUS / 2) * math.sin(spoke_angle)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y))

        # Draw the main interactive node
        node_x = int(self.CENTER[0] + radius * math.cos(angle))
        node_y = int(self.CENTER[1] + radius * math.sin(angle))
        
        # Glow effect for the node
        pygame.gfxdraw.filled_circle(self.screen, node_x, node_y, self.GEAR_NODE_RADIUS, (*color, 50))
        pygame.gfxdraw.filled_circle(self.screen, node_x, node_y, self.GEAR_NODE_RADIUS-3, (*color, 100))
        pygame.gfxdraw.filled_circle(self.screen, node_x, node_y, self.GEAR_NODE_RADIUS-6, color)
        pygame.gfxdraw.aacircle(self.screen, node_x, node_y, self.GEAR_NODE_RADIUS-6, (255, 255, 255))

    def _create_sparks(self):
        angle_mid = self.large_gear_angle + ((self.small_gear_angle - self.large_gear_angle + math.pi) % (2 * math.pi) - math.pi) / 2
        radius_mid = (self.LARGE_GEAR_RADIUS + self.SMALL_GEAR_RADIUS) / 2
        
        px = self.CENTER[0] + radius_mid * math.cos(angle_mid)
        py = self.CENTER[1] + radius_mid * math.sin(angle_mid)

        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(10, 25)
            self.sparks.append([[px, py], vel, life])

    def _update_and_draw_sparks(self):
        for spark in self.sparks:
            spark[0][0] += spark[1][0]
            spark[0][1] += spark[1][1]
            spark[2] -= 1
            
            size = int(max(0, spark[2] / 5))
            if size > 0:
                pygame.draw.circle(self.screen, self.COLOR_SPARK, [int(p) for p in spark[0]], size)
        
        self.sparks = [s for s in self.sparks if s[2] > 0]

    def _render_ui(self):
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        self._draw_text(timer_text, (self.SCREEN_WIDTH - 10, 10), self.font_small, self.COLOR_TEXT, align="topright")
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (10, 10), self.font_small, self.COLOR_TEXT, align="topleft")

        # Magnetism Status
        status_y = self.SCREEN_HEIGHT - 25
        self._render_status("LARGE GEAR (SPACE)", self.large_gear_magnetism, (self.SCREEN_WIDTH * 0.25, status_y))
        self._render_status("SMALL GEAR (SHIFT)", self.small_gear_magnetism, (self.SCREEN_WIDTH * 0.75, status_y))

        # Game Over Message
        if self.game_over:
            if self.collision_occured:
                msg = "COLLISION"
                color = self.COLOR_REPEL
            else:
                msg = "SUCCESS"
                color = self.COLOR_ATTRACT
            self._draw_text(msg, (self.CENTER[0], self.CENTER[1] - 40), self.font_large, color, align="center")

    def _render_status(self, label, magnetism_type, center_pos):
        label_surf = self.font_status.render(label, True, self.COLOR_TEXT)
        label_rect = label_surf.get_rect(center=(center_pos[0], center_pos[1] - 15))
        self.screen.blit(label_surf, label_rect)
        
        states = ["OFF", "ATTRACT", "REPEL"]
        colors = [self.COLOR_GRID, self.COLOR_ATTRACT, self.COLOR_REPEL]
        
        for i, (state, color) in enumerate(zip(states, colors)):
            box_width = 60
            box_height = 20
            x_pos = center_pos[0] - box_width * 1.5 + i * (box_width + 5)
            rect = pygame.Rect(0, 0, box_width, box_height)
            rect.center = (x_pos + box_width/2, center_pos[1] + 15)

            is_active = (i == magnetism_type)
            border_color = color if is_active else self.COLOR_GRID
            pygame.draw.rect(self.screen, border_color, rect, 2)
            
            if is_active:
                inner_rect = rect.inflate(-6, -6)
                pygame.draw.rect(self.screen, color, inner_rect)

            text_surf = self.font_status.render(state, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)

    def _draw_text(self, text, pos, font, color, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "large_gear_angle": self.large_gear_angle,
            "small_gear_angle": self.small_gear_angle,
            "large_gear_magnetism": self.large_gear_magnetism,
            "small_gear_magnetism": self.small_gear_magnetism,
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # --- Manual Play Testing ---
    # This block is not part of the Gymnasium environment and is for local testing.
    # It will be ignored by the validation system.
    # To run, you will need to `pip install pygame`
    try:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        # Un-dummy the video driver for local rendering
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc. depending on your OS
        pygame.display.init()
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Magnetic Gears")
        clock = pygame.time.Clock()

        running = True
        total_reward = 0
        
        # Action state
        action = [0, 0, 0] # move, space, shift

        while running:
            # Handle user input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        action = [0, 0, 0]
                    if event.key == pygame.K_SPACE:
                        action[1] = 1
                    if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                        action[2] = 1
                if event.type == pygame.KEYUP:
                    # Keyup events are not used for toggling in this setup,
                    # but we reset the action to 0 after step to handle single presses.
                    pass

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Reset single-press actions after they are processed
            action[1] = 0
            action[2] = 0

            # Render the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}")
                # Wait for 'r' to reset or 'q' to quit
                wait_for_reset = True
                while wait_for_reset:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            wait_for_reset = False
                            running = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_r:
                                obs, info = env.reset()
                                total_reward = 0
                                action = [0, 0, 0]
                                wait_for_reset = False
                            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                                wait_for_reset = False
                                running = False

            clock.tick(env.FPS)

        env.close()
    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("This might be because you are running in a headless environment.")
        print("The __main__ block is for local testing with a display.")