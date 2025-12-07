import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:44:58.013774
# Source Brief: brief_01885.md
# Brief Index: 1885
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
        "Repair a series of broken, oscillating pipes before time runs out. "
        "Select the correct tool and apply it to the active pipe to score."
    )
    user_guide = (
        "Controls: Use number keys (1-4) or arrow keys to select a tool. "
        "Press space to apply the tool to the active, glowing pipe."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME = 90.0
        self.MAX_STEPS = 2700 # 90 seconds * 30 FPS
        self.NUM_PIPES = 5
        self.NUM_TOOLS = 4
        self.SPEED_MULTIPLIER = 1.2
        self.OSC_AMPLITUDE = 15
        self.OSC_BASE_FREQ = 0.05

        # --- Colors (Industrial/Steampunk Theme) ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_FRAMEWORK = (50, 60, 70)
        self.COLOR_PIPE_INACTIVE = (80, 90, 100)
        self.COLOR_PIPE_REPAIRED = (70, 180, 70)
        self.COLOR_PIPE_ACTIVE = (255, 140, 0)
        self.COLOR_TOOL_SELECTED = (0, 150, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SPARK_SUCCESS = (100, 255, 100)
        self.COLOR_SPARK_FAIL = (255, 50, 50)
        self.TOOL_COLORS = [
            (255, 80, 80),
            (80, 255, 80),
            (80, 80, 255),
            (255, 255, 80),
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_TIME
        self.oscillation_speed_factor = 1.0
        self.selected_tool_index = 0
        self.last_movement_action = 0
        self.previous_space_held = False
        self.pipes = []
        self.active_pipe_index = -1
        self.particles = []
        
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_TIME
        self.oscillation_speed_factor = 1.0
        self.selected_tool_index = 0
        self.last_movement_action = 0
        self.previous_space_held = False
        self.particles = []

        self.pipes = []
        pipe_y_start = 80
        pipe_y_spacing = (self.SCREEN_HEIGHT - 180) // (self.NUM_PIPES - 1)
        
        required_tools = self.np_random.integers(0, self.NUM_TOOLS, size=self.NUM_PIPES)
        
        for i in range(self.NUM_PIPES):
            self.pipes.append({
                "id": i,
                "base_y": pipe_y_start + i * pipe_y_spacing,
                "repaired": False,
                "tool_type": required_tools[i],
                "phase_offset": self.np_random.uniform(0, 2 * math.pi)
            })
        
        self._find_next_active_pipe()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self.steps += 1
        self.time_remaining = max(0, self.time_remaining - 1.0 / self.FPS)

        # 1. Handle Tool Selection
        if movement != 0 and movement != self.last_movement_action:
            new_tool_index = movement - 1 # 1-4 maps to 0-3
            self.selected_tool_index = new_tool_index
            
            if self.active_pipe_index != -1:
                active_pipe = self.pipes[self.active_pipe_index]
                if active_pipe["tool_type"] == self.selected_tool_index:
                    reward += 0.1 # Correct tool selected
                else:
                    reward -= 0.1 # Incorrect tool selected
        self.last_movement_action = movement

        # 2. Handle Tool Application (on rising edge of space bar)
        if space_held and not self.previous_space_held:
            if self.active_pipe_index != -1:
                active_pipe = self.pipes[self.active_pipe_index]
                pipe_pos = self._get_pipe_center(active_pipe)

                if active_pipe["tool_type"] == self.selected_tool_index:
                    # Correct tool applied
                    active_pipe["repaired"] = True
                    reward += 1.0
                    self.score += 1
                    self.oscillation_speed_factor *= self.SPEED_MULTIPLIER
                    self._create_particles(pipe_pos, self.COLOR_SPARK_SUCCESS, 30)
                    # sfx: success_chime.wav
                    self._find_next_active_pipe()
                else:
                    # Incorrect tool applied
                    self._create_particles(pipe_pos, self.COLOR_SPARK_FAIL, 15)
                    # sfx: failure_buzz.wav

        self.previous_space_held = space_held

        # 3. Update Game Logic
        self._update_particles()
        
        # 4. Check for Termination
        terminated = False
        num_repaired = sum(1 for p in self.pipes if p["repaired"])

        if num_repaired == self.NUM_PIPES:
            reward += 100.0 # Win condition
            terminated = True
            self.game_over = True
        elif self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            reward -= 100.0 # Lose condition
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _find_next_active_pipe(self):
        self.active_pipe_index = -1
        for i, pipe in enumerate(self.pipes):
            if not pipe["repaired"]:
                self.active_pipe_index = i
                break

    def _get_pipe_center(self, pipe):
        if pipe["repaired"]:
            y_pos = pipe["base_y"]
        else:
            oscillation = self.OSC_AMPLITUDE * math.sin(
                (self.steps * self.OSC_BASE_FREQ + pipe["phase_offset"]) * self.oscillation_speed_factor
            )
            y_pos = pipe["base_y"] + oscillation
        return (self.SCREEN_WIDTH // 2, int(y_pos))

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

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
            "time_remaining": self.time_remaining,
            "repaired_pipes": sum(1 for p in self.pipes if p["repaired"]),
        }

    def _render_game(self):
        # Render background framework
        pygame.draw.rect(self.screen, self.COLOR_FRAMEWORK, (100, 0, 20, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_FRAMEWORK, (self.SCREEN_WIDTH - 120, 0, 20, self.SCREEN_HEIGHT))

        # Render pipes
        for i, pipe in enumerate(self.pipes):
            x, y = self._get_pipe_center(pipe)
            
            if pipe["repaired"]:
                color = self.COLOR_PIPE_REPAIRED
            elif i == self.active_pipe_index:
                color = self.COLOR_PIPE_ACTIVE
            else:
                color = self.COLOR_PIPE_INACTIVE
            
            # Draw main pipe body with anti-aliasing
            pygame.draw.line(self.screen, color, (110, y), (self.SCREEN_WIDTH - 110, y), 12)
            pygame.gfxdraw.filled_circle(self.screen, 110, y, 8, color)
            pygame.gfxdraw.aacircle(self.screen, 110, y, 8, color)
            pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH - 110, y, 8, color)
            pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH - 110, y, 8, color)

            # Draw tool requirement icon
            tool_color = self.TOOL_COLORS[pipe["tool_type"]]
            pygame.draw.rect(self.screen, tool_color, (self.SCREEN_WIDTH - 100, y - 10, 20, 20))
            pygame.draw.rect(self.screen, self.COLOR_FRAMEWORK, (self.SCREEN_WIDTH - 100, y - 10, 20, 20), 2)


        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 30.0))))
            color = (*p["color"], alpha)
            
            # Create a temporary surface for alpha blending
            particle_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(particle_surf, 2, 2, 2, color)
            self.screen.blit(particle_surf, (int(p["pos"][0]) - 2, int(p["pos"][1]) - 2))


    def _render_ui(self):
        # Render inventory bar
        inventory_y = self.SCREEN_HEIGHT - 60
        inventory_height = 50
        pygame.draw.rect(self.screen, self.COLOR_FRAMEWORK, (0, inventory_y, self.SCREEN_WIDTH, inventory_height))

        slot_width = 80
        slot_spacing = 20
        total_width = self.NUM_TOOLS * slot_width + (self.NUM_TOOLS - 1) * slot_spacing
        start_x = (self.SCREEN_WIDTH - total_width) // 2

        for i in range(self.NUM_TOOLS):
            slot_x = start_x + i * (slot_width + slot_spacing)
            slot_rect = pygame.Rect(slot_x, inventory_y + 5, slot_width, inventory_height - 10)
            
            # Draw tool color
            pygame.draw.rect(self.screen, self.TOOL_COLORS[i], slot_rect.inflate(-10, -10))
            
            # Draw selection highlight
            if i == self.selected_tool_index:
                pygame.draw.rect(self.screen, self.COLOR_TOOL_SELECTED, slot_rect, 4, border_radius=5)
            else:
                pygame.draw.rect(self.screen, self.COLOR_BG, slot_rect, 2, border_radius=5)

            # Draw tool number
            tool_text = self.font_medium.render(str(i + 1), True, self.COLOR_TEXT)
            text_rect = tool_text.get_rect(center=slot_rect.center)
            self.screen.blit(tool_text, text_rect)

        # Render Timer / Pressure Gauge
        gauge_rect = pygame.Rect(self.SCREEN_WIDTH - 40, 20, 20, self.SCREEN_HEIGHT - 100)
        time_ratio = self.time_remaining / self.MAX_TIME
        
        # Gauge color changes from green to red
        gauge_color = (
            int(255 * (1 - time_ratio)),
            int(255 * time_ratio),
            0
        )
        
        pygame.draw.rect(self.screen, self.COLOR_FRAMEWORK, gauge_rect, 2, border_radius=5)
        fill_height = int(gauge_rect.height * time_ratio)
        fill_rect = pygame.Rect(gauge_rect.x, gauge_rect.y + gauge_rect.height - fill_height, gauge_rect.width, fill_height)
        pygame.draw.rect(self.screen, gauge_color, fill_rect, border_bottom_left_radius=5, border_bottom_right_radius=5)

        # Render numerical timer
        timer_text = self.font_medium.render(f"{self.time_remaining:.1f}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(centerx=gauge_rect.centerx, top=gauge_rect.bottom + 5)
        self.screen.blit(timer_text, timer_rect)

        # Render score
        score_text = self.font_large.render(f"Repaired: {self.score}/{self.NUM_PIPES}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(top=10, left=20)
        self.screen.blit(score_text, score_rect)

        # Render Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score == self.NUM_PIPES:
                msg = "SYSTEM REPAIRED"
                color = self.COLOR_PIPE_REPAIRED
            else:
                msg = "PRESSURE CRITICAL"
                color = self.COLOR_SPARK_FAIL
            
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, end_rect)


    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test specific game mechanics
        self.reset()
        initial_speed = self.oscillation_speed_factor
        self.active_pipe_index = 0
        self.selected_tool_index = self.pipes[0]["tool_type"]
        self.step([0, 1, 0]) # Apply correct tool
        assert self.oscillation_speed_factor == initial_speed * self.SPEED_MULTIPLIER
        assert self.pipes[0]["repaired"] == True
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == '__main__':
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Controls: 1, 2, 3, 4 to select tool. Space to apply.
    pygame.display.set_caption("Pipe Repair")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    action = [0, 0, 0] # No-op, release, release
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle key presses for manual control
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: action[0] = 1 # up
                elif event.key == pygame.K_2: action[0] = 2 # down
                elif event.key == pygame.K_3: action[0] = 3 # left
                elif event.key == pygame.K_4: action[0] = 4 # right
                elif event.key == pygame.K_SPACE: action[1] = 1 # space held
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
                    action = [0, 0, 0]
            
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                    action[0] = 0 # no movement
                if event.key == pygame.K_SPACE:
                    action[1] = 0 # space released
    
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished! Total Reward: {total_reward:.2f}, Info: {info}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0
            action = [0, 0, 0]
            
        env.clock.tick(env.FPS)

    env.close()