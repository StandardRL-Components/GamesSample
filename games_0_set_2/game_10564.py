import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:35:12.512131
# Source Brief: brief_00564.md
# Brief Index: 564
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
        "Launch colored neutrons into a rotating nucleus. Match the neutron's color with the nucleus segment to score points and trigger chain reactions."
    )
    user_guide = (
        "Controls: Use ←→ arrows to adjust launch power and ↑↓ arrows to adjust the angle. Press space to launch the neutron."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_RED = (255, 50, 50)
    COLOR_GREEN = (50, 255, 50)
    COLOR_BLUE = (50, 100, 255)
    COLOR_WHITE = (240, 240, 240)
    COLOR_YELLOW = (255, 255, 0)
    COLOR_UI_TEXT = (200, 200, 220)
    
    # Game Parameters
    NET_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    NET_RADIUS = 80
    NET_THICKNESS = 15
    LAUNCH_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 20)
    NEUTRON_RADIUS = 8
    MIN_LAUNCH_POWER = 4.0
    MAX_LAUNCH_POWER = 12.0
    MIN_LAUNCH_ANGLE = math.pi * 1.15
    MAX_LAUNCH_ANGLE = math.pi * 1.85
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Uninitialized state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = "aiming"
        self.launch_angle = 0.0
        self.launch_power = 0.0
        self.prev_space_held = False
        self.neutron = None
        self.neutron_trail = []
        self.next_neutron_color_id = 0
        self.net_rotation = 0.0
        self.net_rotation_speed = 0.0
        self.particles = []
        self.chain_multiplier = 1
        self.last_hit_was_chain = False
        self.score_milestones_achieved = set()
        self.score_pop_timer = 0
        
        self.colors = [self.COLOR_RED, self.COLOR_GREEN, self.COLOR_BLUE]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = "aiming"
        self.launch_angle = math.pi * 1.5
        self.launch_power = (self.MIN_LAUNCH_POWER + self.MAX_LAUNCH_POWER) / 2
        self.prev_space_held = False
        self.neutron = None
        self.neutron_trail = []
        self.next_neutron_color_id = self.np_random.integers(0, 3)
        self.net_rotation = 0.0
        self.net_rotation_speed = 0.01
        self.particles = []
        self.chain_multiplier = 1
        self.last_hit_was_chain = False
        self.score_milestones_achieved = set()
        self.score_pop_timer = 0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self._update_game_state(movement, space_held)
        
        if self.neutron:
            collision_result = self._handle_neutron_movement()
            if collision_result:
                hit_type, multiplier = collision_result
                if hit_type == "chain":
                    reward += 1.0 * multiplier
                    self.score += 1 * multiplier
                    # sfx: chain_reaction.wav
                else: # "sink"
                    reward += 0.1
                    self.score += 1
                    # sfx: sink_neutron.wav
                self.score_pop_timer = 10 # Pop score text for 10 frames
        
        # Milestone rewards
        current_milestone = self.score // 100
        if current_milestone > 0 and current_milestone not in self.score_milestones_achieved:
            reward += current_milestone * 10
            self.score_milestones_achieved.add(current_milestone)

        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_game_state(self, movement, space_held):
        # Update net rotation and difficulty
        self.net_rotation += self.net_rotation_speed
        if self.steps > 0 and self.steps % 500 == 0:
            self.net_rotation_speed += 0.001

        # Update particles
        self._update_particles()
        
        if self.score_pop_timer > 0:
            self.score_pop_timer -= 1

        if self.game_phase == "aiming":
            # Control aiming
            if movement == 1: self.launch_angle += 0.03 # Up
            if movement == 2: self.launch_angle -= 0.03 # Down
            if movement == 4: self.launch_power += 0.1 # Right
            if movement == 3: self.launch_power -= 0.1 # Left
            
            self.launch_angle = np.clip(self.launch_angle, self.MIN_LAUNCH_ANGLE, self.MAX_LAUNCH_ANGLE)
            self.launch_power = np.clip(self.launch_power, self.MIN_LAUNCH_POWER, self.MAX_LAUNCH_POWER)

            # Launch neutron on space press (rising edge)
            if space_held and not self.prev_space_held:
                self._launch_neutron()
                # sfx: launch.wav

        self.prev_space_held = space_held

    def _launch_neutron(self):
        self.game_phase = "flight"
        vel_x = self.launch_power * math.cos(self.launch_angle)
        vel_y = self.launch_power * math.sin(self.launch_angle)
        
        self.neutron = {
            "pos": list(self.LAUNCH_POS),
            "vel": [vel_x, vel_y],
            "color_id": self.next_neutron_color_id,
            "color": self.colors[self.next_neutron_color_id],
        }
        self.neutron_trail = []
        self._create_particles(self.LAUNCH_POS, 15, self.colors[self.next_neutron_color_id], 2.0, 15)
        self.next_neutron_color_id = self.np_random.integers(0, 3)

    def _handle_neutron_movement(self):
        # Update position
        self.neutron["pos"][0] += self.neutron["vel"][0]
        self.neutron["pos"][1] += self.neutron["vel"][1]
        
        # Add to trail
        self.neutron_trail.append(tuple(self.neutron["pos"]))
        if len(self.neutron_trail) > 20:
            self.neutron_trail.pop(0)

        # Check for collision with net
        dist_to_center = math.hypot(self.neutron["pos"][0] - self.NET_POS[0], self.neutron["pos"][1] - self.NET_POS[1])
        
        if self.NET_RADIUS - self.NET_THICKNESS < dist_to_center < self.NET_RADIUS + self.NEUTRON_RADIUS:
            # Collision detected
            hit_angle = math.atan2(self.neutron["pos"][1] - self.NET_POS[1], self.neutron["pos"][0] - self.NET_POS[0])
            normalized_hit_angle = (hit_angle - self.net_rotation) % (2 * math.pi)
            
            segment_angle = 2 * math.pi / 3
            hit_segment_id = int(normalized_hit_angle / segment_angle)
            
            result = None
            if self.neutron["color_id"] == hit_segment_id:
                # Chain reaction
                if self.last_hit_was_chain:
                    self.chain_multiplier += 1
                else:
                    self.chain_multiplier = 2
                self.last_hit_was_chain = True
                result = ("chain", self.chain_multiplier)
                self._create_particles(self.neutron["pos"], 50, self.neutron["color"], 4.0, 40)
            else:
                # Simple sink
                self.chain_multiplier = 1
                self.last_hit_was_chain = False
                result = ("sink", 1)
                self._create_particles(self.neutron["pos"], 20, self.COLOR_WHITE, 1.5, 25)

            self.neutron = None
            self.game_phase = "aiming"
            return result
        
        # Check if off-screen
        nx, ny = self.neutron["pos"]
        if not (0 < nx < self.SCREEN_WIDTH and 0 < ny < self.SCREEN_HEIGHT):
            self.neutron = None
            self.game_phase = "aiming"
            self.chain_multiplier = 1
            self.last_hit_was_chain = False
            # sfx: miss.wav
        
        return None

    def _create_particles(self, pos, count, color, max_speed, lifespan):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos), "vel": vel, "life": lifespan, "max_life": lifespan, "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.98 # friction
            p["vel"][1] *= 0.98
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            size = int(3 * (p["life"] / p["max_life"]))
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p["pos"][0]), int(p["pos"][1])), size)

        # Render nucleus net with glow
        segment_angle = 2 * math.pi / 3
        for i in range(3):
            start_angle = self.net_rotation + i * segment_angle
            end_angle = start_angle + segment_angle
            color = self.colors[i]
            
            # Glow effect
            for j in range(8):
                alpha = 80 - j * 10
                glow_color = (*color, alpha)
                radius = self.NET_RADIUS + j
                thickness = self.NET_THICKNESS + j * 2
                rect = pygame.Rect(self.NET_POS[0] - radius, self.NET_POS[1] - radius, radius * 2, radius * 2)
                pygame.draw.arc(self.screen, glow_color, rect, -end_angle, -start_angle, thickness)
        
        # Render aiming UI
        if self.game_phase == "aiming":
            # Aiming line
            end_x = self.LAUNCH_POS[0] + 150 * math.cos(self.launch_angle)
            end_y = self.LAUNCH_POS[1] + 150 * math.sin(self.launch_angle)
            self._draw_dashed_line(self.LAUNCH_POS, (end_x, end_y), self.COLOR_WHITE)
            
            # Power bar
            power_ratio = (self.launch_power - self.MIN_LAUNCH_POWER) / (self.MAX_LAUNCH_POWER - self.MIN_LAUNCH_POWER)
            power_bar_width = 100
            power_bar_height = 10
            bar_x = self.LAUNCH_POS[0] - power_bar_width / 2
            bar_y = self.LAUNCH_POS[1] + 10
            pygame.draw.rect(self.screen, (50, 50, 80), (bar_x, bar_y, power_bar_width, power_bar_height), 1)
            pygame.draw.rect(self.screen, self.COLOR_YELLOW, (bar_x, bar_y, power_bar_width * power_ratio, power_bar_height))
            
            # Ready neutron
            self._draw_glowing_circle(self.LAUNCH_POS, self.NEUTRON_RADIUS, self.colors[self.next_neutron_color_id])

        # Render in-flight neutron
        if self.neutron:
            # Trail
            for i, pos in enumerate(self.neutron_trail):
                alpha = int(255 * (i / len(self.neutron_trail)))
                color = (*self.neutron["color"], alpha)
                radius = int(self.NEUTRON_RADIUS * 0.5 * (i / len(self.neutron_trail)))
                if radius > 0:
                    pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), radius)
            # Neutron
            self._draw_glowing_circle(self.neutron["pos"], self.NEUTRON_RADIUS, self.neutron["color"])
            
    def _draw_glowing_circle(self, pos, radius, color):
        x, y = int(pos[0]), int(pos[1])
        # Glow
        for i in range(radius, 0, -2):
            alpha = int(100 * (1 - i / radius))
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius + i, (*color, alpha))
        # Main circle
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def _draw_dashed_line(self, start, end, color, dash_length=5):
        x1, y1 = start
        x2, y2 = end
        dx, dy = x2 - x1, y2 - y1
        distance = math.hypot(dx, dy)
        if distance == 0: return
        dashes = int(distance / dash_length)
        
        for i in range(dashes):
            start_pos = (x1 + dx * i / dashes, y1 + dy * i / dashes)
            end_pos = (x1 + dx * (i + 0.5) / dashes, y1 + dy * (i + 0.5) / dashes)
            pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

    def _render_ui(self):
        # Score display
        score_size = 24 + self.score_pop_timer
        score_font = pygame.font.SysFont("Consolas", score_size, bold=True)
        score_text = score_font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topleft=(10, 10))
        self.screen.blit(score_text, score_rect)

        # Multiplier display
        if self.chain_multiplier > 1:
            mult_text = self.font_main.render(f"x{self.chain_multiplier}", True, self.COLOR_YELLOW)
            self.screen.blit(mult_text, (score_rect.right + 10, 10))

        # Next neutron display
        next_text = self.font_small.render("NEXT:", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (10, self.SCREEN_HEIGHT - 30))
        self._draw_glowing_circle((70, self.SCREEN_HEIGHT - 22), self.NEUTRON_RADIUS + 2, self.colors[self.next_neutron_color_id])
        
        # Steps display
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        steps_rect = steps_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(steps_text, steps_rect)

    def close(self):
        pygame.quit()
    
    def render(self):
        # This method is not used by the environment's step/reset but can be useful for external rendering.
        return self._get_observation()

    def validate_implementation(self):
        # This method was in the original code for self-checking, but is not part of the standard Gym API.
        # It's good practice for development but can be removed or commented out in a final version.
        try:
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
            assert isinstance(trunc, bool)
            assert isinstance(info, dict)
            
            print("✓ Implementation validated successfully")
        except AssertionError as e:
            print(f"✗ Implementation validation failed: {e}")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Un-dummy the video driver to see the game window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for manual play
    pygame.display.set_caption("Neutron Netball")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Controls ---
        movement = 0 # none
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Run at 60 FPS for smooth human gameplay
        
    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()