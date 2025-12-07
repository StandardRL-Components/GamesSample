import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:01:37.072347
# Source Brief: brief_00166.md
# Brief Index: 166
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
        "Manipulate tectonic plates to reshape the earth. Set plate speeds and trigger collisions "
        "to build mountains and valleys, aiming to reach a target score before you run out of turns."
    )
    user_guide = (
        "Use ↑/↓ to select a plate and ←/→ to adjust its time warp. "
        "Press space to flip the direction of geological uplift. "
        "Press shift to end your turn and simulate the plate collisions."
    )
    auto_advance = False

    # --- Class-level attributes for state that persists across resets ---
    # Unlocked plate types can be expanded upon winning
    UNLOCKED_PLATE_TYPES = {
        "continental": {"mass": 1.0, "color": (210, 180, 140)}, # Tan
        "oceanic": {"mass": 1.5, "color": (70, 130, 180)},     # Steel Blue
    }
    # Track wins to increase difficulty
    _last_episode_was_win = False
    _base_target_score = 50.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width = 640
        self.screen_height = 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Visuals & Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_BG_ACCENT = (25, 25, 45)
        self.COLOR_FAULTLINE = (40, 40, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_ACCENT = (100, 255, 255) # Cyan
        self.COLOR_INDICATOR = (255, 200, 80) # Amber
        self.COLOR_GRAVITY_UP = (100, 255, 100)
        self.COLOR_GRAVITY_DOWN = (255, 100, 100)
        
        self.font_main = pygame.font.SysFont("Consolas", 18)
        self.font_accent = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # Game constants
        self.max_turns = 50
        self.fault_line_y = self.screen_height // 2
        self.time_factors = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]

        # Initialize state variables (will be properly set in reset)
        self.plates = []
        self.particles = []
        self.terrain_heights = np.zeros(self.screen_width)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.turns_remaining = 0
        self.target_score = 0
        self.selected_plate_idx = 0
        self.gravity_up = True
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Initialize state
        # self.reset() is called by the wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Update target score based on previous episode's outcome
        if GameEnv._last_episode_was_win:
            GameEnv._base_target_score *= 1.10
        self.target_score = int(GameEnv._base_target_score)
        GameEnv._last_episode_was_win = False # Reset for the new episode

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.turns_remaining = self.max_turns
        self.selected_plate_idx = 0
        self.gravity_up = True
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        
        # Initialize terrain
        self.terrain_heights = np.zeros(self.screen_width)
        
        # Initialize plates
        self._initialize_plates()
        
        return self._get_observation(), self._get_info()

    def _initialize_plates(self):
        self.plates = []
        num_plates = self.np_random.integers(3, 6)
        occupied_zones = []

        for i in range(num_plates):
            plate_type_name = self.np_random.choice(list(GameEnv.UNLOCKED_PLATE_TYPES.keys()))
            plate_spec = GameEnv.UNLOCKED_PLATE_TYPES[plate_type_name]
            
            size = self.np_random.integers(40, 70)
            while True:
                pos_x = self.np_random.integers(size, self.screen_width - size)
                # Ensure no overlap
                is_overlapping = any(abs(pos_x - ox) < (size + o_size) / 2 for ox, o_size in occupied_zones)
                if not is_overlapping:
                    break
            
            occupied_zones.append((pos_x, size))
            
            self.plates.append({
                "pos": pygame.Vector2(pos_x, self.fault_line_y),
                "vel": self.np_random.uniform(-1.0, 1.0),
                "size": size,
                "mass": plate_spec["mass"] * (size / 50.0), # Mass scales with size
                "color": plate_spec["color"],
                "time_factor_idx": 2, # Default to 1.0x
            })
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        turn_ended = False
        if shift_press:
            turn_reward = self._end_turn()
            reward += turn_reward
            turn_ended = True
        else:
            self._handle_planning_actions(movement, space_press)

        self._update_particles()
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.score >= self.target_score:
                reward += 100
                GameEnv._last_episode_was_win = True
            elif self.turns_remaining <= 0:
                reward -= 100
        
        # Clamp reward
        reward = np.clip(reward, -100, 100)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_planning_actions(self, movement, space_press):
        if not self.plates: return

        # Cycle through plates
        if movement == 1: # Up
            self.selected_plate_idx = (self.selected_plate_idx - 1) % len(self.plates)
        elif movement == 2: # Down
            self.selected_plate_idx = (self.selected_plate_idx + 1) % len(self.plates)
        
        # Adjust time factor for selected plate
        plate = self.plates[self.selected_plate_idx]
        if movement == 3: # Left
            plate["time_factor_idx"] = max(0, plate["time_factor_idx"] - 1)
        elif movement == 4: # Right
            plate["time_factor_idx"] = min(len(self.time_factors) - 1, plate["time_factor_idx"] + 1)
        
        # Flip gravity
        if space_press:
            self.gravity_up = not self.gravity_up
            # sfx: gravity_flip.wav

    def _end_turn(self):
        self.turns_remaining -= 1
        
        # 1. Update plate positions
        for p in self.plates:
            time_factor = self.time_factors[p["time_factor_idx"]]
            p["pos"].x += p["vel"] * time_factor * 5.0 # Scaled for visibility

        # 2. Handle collisions
        collisions = 0
        total_energy = 0
        
        # Sort plates by position for efficient collision checking
        self.plates.sort(key=lambda p: p["pos"].x)

        for i in range(len(self.plates)):
            # Wall collisions
            p1 = self.plates[i]
            if p1["pos"].x - p1["size"] / 2 < 0:
                p1["pos"].x = p1["size"] / 2
                p1["vel"] *= -0.8 # Dampened bounce
            if p1["pos"].x + p1["size"] / 2 > self.screen_width:
                p1["pos"].x = self.screen_width - p1["size"] / 2
                p1["vel"] *= -0.8 # Dampened bounce

            # Inter-plate collisions
            if i < len(self.plates) - 1:
                p2 = self.plates[i+1]
                dist = p2["pos"].x - p1["pos"].x
                min_dist = (p1["size"] + p2["size"]) / 2
                
                if dist < min_dist:
                    collisions += 1
                    # sfx: earthquake_rumble.wav

                    # Resolve overlap
                    overlap = min_dist - dist
                    p1["pos"].x -= overlap / 2
                    p2["pos"].x += overlap / 2

                    # Collision physics (1D elastic)
                    m1, m2 = p1["mass"], p2["mass"]
                    v1, v2 = p1["vel"], p2["vel"]
                    new_v1 = ((m1 - m2) / (m1 + m2)) * v1 + (2 * m2 / (m1 + m2)) * v2
                    new_v2 = (2 * m1 / (m1 + m2)) * v1 + ((m2 - m1) / (m1 + m2)) * v2
                    p1["vel"] = new_v1
                    p2["vel"] = new_v2
                    
                    # Calculate geological impact
                    collision_pos = (p1["pos"].x + p2["pos"].x) / 2
                    energy = abs(v1 - v2) * (m1 + m2)
                    total_energy += energy
                    self._apply_geological_change(collision_pos, energy)
                    self._add_shockwave(collision_pos, energy)

        # 3. Calculate score and reward
        old_score = self.score
        self.score = int(np.sum(np.abs(self.terrain_heights)))
        score_delta = self.score - old_score

        reward = 0
        if collisions > 0:
            reward += collisions * 1.0 # Event-based reward
        reward += score_delta * 0.1 # Continuous feedback

        return reward

    def _apply_geological_change(self, pos, energy):
        spread = max(20, energy * 20)
        magnitude = energy * 5
        direction = 1 if self.gravity_up else -1
        
        x = np.arange(self.screen_width)
        # Gaussian distribution for the "mountain"
        change = magnitude * np.exp(-((x - pos) ** 2) / (2 * spread ** 2))
        self.terrain_heights += change * direction
        self.terrain_heights = np.clip(self.terrain_heights, -self.fault_line_y + 20, self.fault_line_y - 20)

    def _add_shockwave(self, pos_x, energy):
        # sfx: shockwave_burst.wav
        num_particles = min(10, int(energy * 2))
        for _ in range(num_particles):
            self.particles.append({
                "pos": pygame.Vector2(pos_x, self.fault_line_y),
                "radius": self.np_random.uniform(1, 5),
                "max_radius": max(30, energy * 30),
                "speed": self.np_random.uniform(1, 3),
                "life": 1.0, # 100% life
                "color": (255, 255, self.np_random.integers(150, 255))
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["radius"] += p["speed"]
            p["life"] -= 0.02
            if p["radius"] > p["max_radius"]:
                p["life"] -= 0.1 # Fade faster after reaching max radius

    def _check_termination(self):
        return self.turns_remaining <= 0 or self.score >= self.target_score or self.steps >= 1000

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
            "turns_remaining": self.turns_remaining,
            "target_score": self.target_score
        }

    def _render_game(self):
        # Background gradient and fault line
        pygame.draw.rect(self.screen, self.COLOR_BG_ACCENT, (0, self.fault_line_y - 25, self.screen_width, 50))
        pygame.draw.rect(self.screen, self.COLOR_FAULTLINE, (0, self.fault_line_y - 30, self.screen_width, 60))
        
        self._render_terrain()
        self._render_particles()
        self._render_plates()

    def _render_terrain(self):
        baseline_y = self.fault_line_y
        points = [(0, baseline_y)]
        for x, y_offset in enumerate(self.terrain_heights):
            points.append((x, baseline_y + y_offset))
        points.append((self.screen_width, baseline_y))
        points.append((self.screen_width, self.screen_height))
        points.append((0, self.screen_height))

        # Create a gradient for the terrain
        max_h = np.max(np.abs(self.terrain_heights)) if np.any(self.terrain_heights) else 1
        for x in range(self.screen_width):
            h = self.terrain_heights[x]
            # Color based on height, from blue to brown to white
            lerp_val = np.clip(abs(h) / max(1, self.fault_line_y / 2), 0, 1)
            if lerp_val < 0.3:
                color = (40, 40, 80 + lerp_val * 100)
            elif lerp_val < 0.7:
                color = (80 + (lerp_val - 0.3) * 100, 60, 40)
            else:
                color = (180 + (lerp_val - 0.7) * 75, 180 + (lerp_val - 0.7) * 75, 180 + (lerp_val - 0.7) * 75)
            
            start_y = int(baseline_y)
            end_y = int(baseline_y + h)
            if start_y > end_y: start_y, end_y = end_y, start_y
            pygame.draw.line(self.screen, color, (x, start_y), (x, end_y))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, int(p["life"] * 255))
            color = (*p["color"], alpha)
            pygame.gfxdraw.aacircle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color)

    def _render_plates(self):
        for i, plate in enumerate(self.plates):
            pos = (int(plate["pos"].x), int(plate["pos"].y))
            size = int(plate["size"] / 2)
            rect = pygame.Rect(pos[0] - size, pos[1] - size, plate["size"], plate["size"])
            
            # Plate body
            pygame.draw.rect(self.screen, plate["color"], rect, border_radius=8)
            
            # Selection highlight
            if i == self.selected_plate_idx:
                # Glow effect
                for j in range(5, 0, -1):
                    glow_alpha = 30 - j * 5
                    glow_color = (*self.COLOR_INDICATOR, glow_alpha)
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size + j*2, glow_color)
                
                pygame.gfxdraw.arc(self.screen, pos[0], pos[1], size + 5, 0, 360, self.COLOR_INDICATOR)

                # Time factor display
                time_factor = self.time_factors[plate["time_factor_idx"]]
                tf_text = f"{time_factor}x"
                text_surf = self.font_small.render(tf_text, True, self.COLOR_TEXT)
                text_rect = text_surf.get_rect(center=(pos[0], pos[1] - size - 15))
                self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score and Turns
        score_text = f"SCORE: {self.score}"
        target_text = f"TARGET: {self.target_score}"
        turns_text = f"TURNS: {self.turns_remaining}"
        
        score_surf = self.font_accent.render(score_text, True, self.COLOR_TEXT_ACCENT)
        target_surf = self.font_main.render(target_text, True, self.COLOR_TEXT)
        turns_surf = self.font_main.render(turns_text, True, self.COLOR_TEXT)
        
        self.screen.blit(score_surf, (15, 10))
        self.screen.blit(target_surf, (15, 40))
        self.screen.blit(turns_surf, (15, 60))

        # Gravity Indicator
        grav_text = "GRAVITY"
        grav_surf = self.font_main.render(grav_text, True, self.COLOR_TEXT)
        self.screen.blit(grav_surf, (self.screen_width - grav_surf.get_width() - 15, 10))
        
        arrow_color = self.COLOR_GRAVITY_UP if self.gravity_up else self.COLOR_GRAVITY_DOWN
        arrow_points = [(self.screen_width - 55, 45), (self.screen_width - 45, 35), (self.screen_width - 35, 45)] if self.gravity_up else \
                       [(self.screen_width - 55, 35), (self.screen_width - 45, 45), (self.screen_width - 35, 35)]
        pygame.draw.polygon(self.screen, arrow_color, arrow_points)
        pygame.draw.lines(self.screen, arrow_color, False, arrow_points, 2)

        # Controls Help
        help_text = "[UP/DOWN] Select Plate | [LEFT/RIGHT] Time Warp | [SPACE] Flip Gravity | [SHIFT] End Turn"
        help_surf = self.font_small.render(help_text, True, self.COLOR_TEXT)
        self.screen.blit(help_surf, (10, self.screen_height - help_surf.get_height() - 5))
    
    def close(self):
        pygame.quit()

# Example usage:
if __name__ == "__main__":
    # This block is for human play and is not part of the gym environment API.
    # It will not be executed by the test suite.
    # For this to run, you need to unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Tectonic Time Warp")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    running = True
    
    # Remove the `validate_implementation` call as it's not needed for the final code
    # and was just for internal verification in the original script.
    
    last_action_time = pygame.time.get_ticks()

    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Only process actions periodically to match the turn-based nature
        now = pygame.time.get_ticks()
        if now - last_action_time > 150: # Allow an action every 150ms
            keys = pygame.key.get_pressed()
            action_taken = False
            if keys[pygame.K_UP]: 
                movement = 1
                action_taken = True
            elif keys[pygame.K_DOWN]: 
                movement = 2
                action_taken = True
            elif keys[pygame.K_LEFT]: 
                movement = 3
                action_taken = True
            elif keys[pygame.K_RIGHT]: 
                movement = 4
                action_taken = True
            
            if keys[pygame.K_SPACE]: 
                space = 1
                action_taken = True
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: 
                shift = 1
                action_taken = True

            if action_taken:
                action = [movement, space, shift]
                obs, reward, terminated, truncated, info = env.step(action)
                last_action_time = now

                if terminated:
                    print(f"Episode finished. Final Score: {info['score']}. Target: {info['target_score']}")
                    # Display final state
                    surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    pygame.time.wait(2000) # Pause for 2 seconds
                    obs, info = env.reset()
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit to 30 FPS

    env.close()