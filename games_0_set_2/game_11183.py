import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:46:11.020432
# Source Brief: brief_01183.md
# Brief Index: 1183
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
        "Control deep-sea robotic arms to harvest valuable resources erupting from hydrothermal vents. "
        "Use a chain reaction to pull in clusters of resources for bonus points."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the active arm. "
        "Press 'shift' to switch between arms and 'space' to activate a chain reaction pulse."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_W, SCREEN_H = 640, 400
    GRID_W, GRID_H = 32, 20
    CELL_SIZE = 20
    MAX_STEPS = 2000
    FPS = 30

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_ROCK = (25, 35, 55)
    COLOR_VENT = (60, 40, 30)
    COLOR_VENT_HOT = (255, 100, 0)
    COLOR_RESOURCE = (50, 255, 150)
    COLOR_ARM_BODY = (150, 160, 180)
    COLOR_ARM_JOINT = (120, 130, 150)
    COLOR_ARM_ACTIVE_GLOW = (255, 255, 255)
    COLOR_REACTION_PULSE = (100, 200, 255)
    COLOR_TEXT = (220, 220, 240)
    
    # Game Parameters
    ARM_MOVE_COOLDOWN = 3
    CHAIN_REACTION_DURATION = 60
    CHAIN_REACTION_COOLDOWN = 120
    CHAIN_REACTION_RADIUS = 80
    RESOURCE_COLLECT_RADIUS = 25
    
    INITIAL_VENT_TIMER_RANGE = (100, 200)
    INITIAL_RESOURCE_COUNT_RANGE = (8, 15)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18, bold=True)
        
        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.arms = []
        self.active_arm_idx = 0
        self.vents = []
        self.resources = []
        self.particles = []
        
        self.last_space_held = False
        self.last_shift_held = False
        self.score_milestone = 0
        
        self.bg_plankton = []
        
        # self.validate_implementation() # This can be removed for submission

    def _grid_to_pixel(self, grid_pos):
        x = (grid_pos[0] + 0.5) * self.CELL_SIZE
        y = (grid_pos[1] + 0.5) * self.CELL_SIZE
        return int(x), int(y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.score_milestone = 0
        
        self.last_space_held = False
        self.last_shift_held = False

        self.arms = [
            {
                "pos": [self.GRID_W // 4, self.GRID_H // 2],
                "base_pos": (self.SCREEN_W * 0.2, self.SCREEN_H - 10),
                "move_cooldown": 0,
                "chain_reaction_timer": 0,
                "chain_reaction_cooldown": 0,
            },
            {
                "pos": [self.GRID_W * 3 // 4, self.GRID_H // 2],
                "base_pos": (self.SCREEN_W * 0.8, self.SCREEN_H - 10),
                "move_cooldown": 0,
                "chain_reaction_timer": 0,
                "chain_reaction_cooldown": 0,
            },
        ]
        self.active_arm_idx = 0

        self.vents = []
        for i in range(3):
            self.vents.append({
                "pos": ((self.SCREEN_W / 4) * (i + 1), self.SCREEN_H - 20),
                "timer": self.np_random.integers(self.INITIAL_VENT_TIMER_RANGE[0], self.INITIAL_VENT_TIMER_RANGE[1]),
                "eruption_anim": 0,
            })

        self.resources = []
        self.particles = []
        
        if not self.bg_plankton:
             for _ in range(100):
                self.bg_plankton.append([
                    self.np_random.random() * self.SCREEN_W,
                    self.np_random.random() * self.SCREEN_H,
                    self.np_random.random() * 1.5 + 0.5 # speed
                ])
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game State ---
        self._handle_input(movement, space_held, shift_held)
        reward += self._update_game_logic()
        
        terminated = self.steps >= self.MAX_STEPS
        
        # --- Calculate Step Reward ---
        # Collection rewards are handled in _update_game_logic
        # Chain reaction trigger reward
        if space_held and not self.last_space_held:
            active_arm = self.arms[self.active_arm_idx]
            if active_arm["chain_reaction_cooldown"] <= 0:
                reward += 1.0
                # sfx: chain reaction activation sound
        
        # Score milestone reward
        new_milestone = int(self.score // 100)
        if new_milestone > self.score_milestone:
            reward += 10.0 * (new_milestone - self.score_milestone)
            self.score_milestone = new_milestone
            # sfx: milestone achievement sound

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        active_arm = self.arms[self.active_arm_idx]
        
        # Arm movement
        if active_arm["move_cooldown"] <= 0 and movement != 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            new_x = np.clip(active_arm["pos"][0] + dx, 0, self.GRID_W - 1)
            new_y = np.clip(active_arm["pos"][1] + dy, 0, self.GRID_H - 1)
            
            if new_x != active_arm["pos"][0] or new_y != active_arm["pos"][1]:
                active_arm["pos"] = [new_x, new_y]
                active_arm["move_cooldown"] = self.ARM_MOVE_COOLDOWN
                # sfx: arm movement servo sound
        
        # Switch active arm
        if shift_held and not self.last_shift_held:
            self.active_arm_idx = (self.active_arm_idx + 1) % len(self.arms)
            # sfx: arm switch beep
            
        # Activate chain reaction
        if space_held and not self.last_space_held:
            if active_arm["chain_reaction_cooldown"] <= 0:
                active_arm["chain_reaction_timer"] = self.CHAIN_REACTION_DURATION
                active_arm["chain_reaction_cooldown"] = self.CHAIN_REACTION_COOLDOWN

    def _update_game_logic(self):
        # --- Update Cooldowns ---
        for arm in self.arms:
            if arm["move_cooldown"] > 0: arm["move_cooldown"] -= 1
            if arm["chain_reaction_timer"] > 0: arm["chain_reaction_timer"] -= 1
            if arm["chain_reaction_cooldown"] > 0: arm["chain_reaction_cooldown"] -= 1
            
        # --- Update Vents and Spawn Resources ---
        self._update_vents()
        
        # --- Update Resources ---
        collection_reward = self._update_resources()
        
        # --- Update Particles ---
        self._update_particles()

        return collection_reward
        
    def _update_difficulty(self):
        difficulty_level = self.steps // 500
        
        # Scale vent eruption time
        scaling_factor = max(0.5, 1.0 - (difficulty_level * 0.1))
        vent_timer_range = (
            int(self.INITIAL_VENT_TIMER_RANGE[0] * scaling_factor),
            int(self.INITIAL_VENT_TIMER_RANGE[1] * scaling_factor)
        )
        
        # Scale resource count
        resource_scaling_factor = 1.0 + (difficulty_level * 0.1)
        resource_count_range = (
            int(self.INITIAL_RESOURCE_COUNT_RANGE[0] * resource_scaling_factor),
            int(self.INITIAL_RESOURCE_COUNT_RANGE[1] * resource_scaling_factor)
        )
        return vent_timer_range, resource_count_range

    def _update_vents(self):
        vent_timer_range, resource_count_range = self._update_difficulty()
        
        for vent in self.vents:
            if vent["eruption_anim"] > 0:
                vent["eruption_anim"] -= 1

            vent["timer"] -= 1
            if vent["timer"] <= 0:
                vent["timer"] = self.np_random.integers(vent_timer_range[0], vent_timer_range[1])
                vent["eruption_anim"] = 20
                
                num_resources = self.np_random.integers(resource_count_range[0], resource_count_range[1])
                # sfx: vent eruption rumble
                
                for _ in range(num_resources):
                    angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                    speed = self.np_random.uniform(1.5, 3.0)
                    vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
                    self.resources.append({
                        "pos": pygame.Vector2(vent["pos"]),
                        "vel": vel,
                        "lifetime": self.np_random.integers(120, 180),
                    })
                
                for _ in range(30): # Eruption particles
                    angle = self.np_random.uniform(math.pi, math.pi * 2)
                    speed = self.np_random.uniform(1, 4)
                    vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
                    self.particles.append({
                        "pos": pygame.Vector2(vent["pos"][0] + self.np_random.uniform(-5, 5), vent["pos"][1]),
                        "vel": vel,
                        "lifetime": self.np_random.integers(20, 40),
                        "max_lifetime": 40,
                        "radius": self.np_random.uniform(1, 4),
                        "color": self.COLOR_VENT_HOT
                    })

    def _update_resources(self):
        collection_reward = 0.0
        active_arm_obj = self.arms[self.active_arm_idx]
        active_arm_pos = pygame.Vector2(self._grid_to_pixel(active_arm_obj["pos"]))
        
        remaining_resources = []
        for res in self.resources:
            res["pos"] += res["vel"]
            res["vel"] *= 0.98  # Water resistance
            res["vel"].y -= 0.03 # Buoyancy
            res["lifetime"] -= 1
            
            dist_to_arm = res["pos"].distance_to(active_arm_pos)
            
            # Magnetic pull from active arm
            if dist_to_arm < self.CHAIN_REACTION_RADIUS + 20:
                pull_vec = (active_arm_pos - res["pos"]).normalize() * 0.5
                res["vel"] += pull_vec

            if dist_to_arm < self.RESOURCE_COLLECT_RADIUS:
                # sfx: resource collection tick
                self.score += 1
                collection_reward += 0.1
                
                if active_arm_obj["chain_reaction_timer"] > 0:
                    self.score += 5 # Bonus value
                    collection_reward += 0.5
                    # sfx: bonus collection tick
                
                # Collection particle effect
                for _ in range(5):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 3)
                    vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
                    self.particles.append({
                        "pos": pygame.Vector2(res["pos"]), "vel": vel,
                        "lifetime": self.np_random.integers(10, 20), "max_lifetime": 20,
                        "radius": self.np_random.uniform(1, 2.5), "color": self.COLOR_RESOURCE
                    })
            elif res["lifetime"] > 0:
                remaining_resources.append(res)
                
        self.resources = remaining_resources
        return collection_reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifetime"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
            p["vel"] *= 0.95
    
    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        self._render_background_elements()

        # --- Game Objects ---
        self._render_vents()
        self._render_particles(self.resources, self.COLOR_RESOURCE, is_resource=True)
        self._render_particles(self.particles)
        self._render_arms()

        # --- UI ---
        self._render_ui()

    def _render_background_elements(self):
        # Plankton
        for p in self.bg_plankton:
            p[0] = (p[0] - p[2] * 0.1) % self.SCREEN_W
            p[1] = (p[1] - p[2] * 0.2) % self.SCREEN_H
            color_val = int(30 + p[2] * 10)
            pygame.draw.circle(self.screen, (color_val, color_val+10, color_val+20), (int(p[0]), int(p[1])), 1)
        
        # Seafloor
        pygame.draw.rect(self.screen, self.COLOR_ROCK, (0, self.SCREEN_H - 25, self.SCREEN_W, 25))
        for i in range(15):
            x = i * 50 + (self.steps/5 % 50) - 25
            h = self.np_random.uniform(5, 15)
            pygame.draw.polygon(self.screen, self.COLOR_BG, [
                (x, self.SCREEN_H - 25), (x+25, self.SCREEN_H - 25 - h), (x+50, self.SCREEN_H - 25)
            ])


    def _render_vents(self):
        for vent in self.vents:
            pos_x, pos_y = int(vent["pos"][0]), int(vent["pos"][1])
            pygame.draw.polygon(self.screen, self.COLOR_VENT, [
                (pos_x - 15, self.SCREEN_H), (pos_x - 10, pos_y),
                (pos_x + 10, pos_y), (pos_x + 15, self.SCREEN_H)
            ])
            if vent["eruption_anim"] > 0:
                alpha = int(255 * (vent["eruption_anim"] / 20))
                self._draw_glowing_circle(self.screen, pos_x, pos_y, 10, self.COLOR_VENT_HOT, alpha)

    def _render_arms(self):
        for i, arm in enumerate(self.arms):
            is_active = i == self.active_arm_idx
            
            # Draw arm base
            base_x, base_y = int(arm["base_pos"][0]), int(arm["base_pos"][1])
            pygame.draw.rect(self.screen, self.COLOR_ARM_JOINT, (base_x - 15, base_y - 10, 30, 15))
            pygame.draw.circle(self.screen, self.COLOR_ARM_JOINT, (base_x, base_y - 10), 10)

            # Draw arm segments
            head_pos = self._grid_to_pixel(arm["pos"])
            mid_pos = ( (base_x + head_pos[0]) // 2, (base_y - 10 + head_pos[1]) // 2 - 50 )
            
            pygame.draw.line(self.screen, self.COLOR_ARM_BODY, (base_x, base_y - 10), mid_pos, 8)
            pygame.draw.line(self.screen, self.COLOR_ARM_BODY, mid_pos, head_pos, 6)
            
            pygame.draw.circle(self.screen, self.COLOR_ARM_JOINT, (base_x, base_y - 10), 5)
            pygame.draw.circle(self.screen, self.COLOR_ARM_JOINT, mid_pos, 6)

            # Draw collector head
            if is_active:
                glow_alpha = 100 + int(155 * (math.sin(self.steps / 10) * 0.5 + 0.5))
                self._draw_glowing_circle(self.screen, head_pos[0], head_pos[1], self.RESOURCE_COLLECT_RADIUS, self.COLOR_ARM_ACTIVE_GLOW, glow_alpha)
            
            pygame.draw.circle(self.screen, self.COLOR_ARM_BODY, head_pos, 12)
            pygame.draw.circle(self.screen, (255, 255, 255) if is_active else self.COLOR_ARM_JOINT, head_pos, 5)

            # Chain reaction visual
            if arm["chain_reaction_timer"] > 0:
                progress = 1.0 - (arm["chain_reaction_timer"] / self.CHAIN_REACTION_DURATION)
                current_radius = int(self.CHAIN_REACTION_RADIUS * math.sin(progress * math.pi))
                alpha = int(200 * (1 - progress))
                self._draw_glowing_circle(self.screen, head_pos[0], head_pos[1], current_radius, self.COLOR_REACTION_PULSE, alpha)

    def _render_particles(self, particle_list, default_color=None, is_resource=False):
        for p in particle_list:
            pos = (int(p["pos"].x), int(p["pos"].y))
            
            if is_resource:
                lifetime_frac = p["lifetime"] / 180.0
                radius = 3 + lifetime_frac * 2
                alpha = int(255 * min(1.0, lifetime_frac * 2))
                self._draw_glowing_circle(self.screen, pos[0], pos[1], int(radius), default_color, alpha)
            else:
                lifetime_frac = p["lifetime"] / p["max_lifetime"]
                radius = p["radius"] * lifetime_frac
                if radius > 1:
                    pygame.draw.circle(self.screen, p["color"], pos, int(radius))

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font.render(f"TIME: {time_left:.1f}s", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_W - time_text.get_width() - 10, 10))

    def _draw_glowing_circle(self, surface, x, y, radius, color, alpha):
        if radius <= 0: return
        
        # Create a temporary surface for the glow
        temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        
        # Draw multiple circles with decreasing alpha for the glow effect
        for i in range(int(radius), 0, -2):
            glow_alpha = int(alpha * (1 - (i / radius))**2)
            if glow_alpha > 0:
                pygame.gfxdraw.aacircle(temp_surf, radius, radius, i, (*color, glow_alpha))
        
        surface.blit(temp_surf, (x - radius, y - radius))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space after reset
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Example Usage ---
    # This section is for local testing and will not be run by the evaluator.
    # To use it, you'll need to remove the "dummy" video driver setting.
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup for manual play
    pygame.display.set_caption("Hydrothermal Harvest")
    screen = pygame.display.set_mode((env.SCREEN_W, env.SCREEN_H))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()