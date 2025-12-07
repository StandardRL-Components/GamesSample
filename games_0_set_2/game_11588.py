import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:12:24.161887
# Source Brief: brief_01588.md
# Brief Index: 1588
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a sci-fi arcade game.
    The player launches different types of neutrons at a central nucleus
    to trigger chain reactions and score points.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    game_description = (
        "Launch different types of neutrons at a central nucleus to trigger chain reactions and score points. "
        "Unlock more powerful neutrons as your score increases."
    )
    user_guide = (
        "Use ↑↓ arrow keys to aim the launcher. Use ←→ arrow keys to select a neutron type. Press space to launch."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1800  # 30 seconds at 60 FPS
        self.INITIAL_TARGET_SCORE = 500
        self.TARGET_SCORE_INCREMENT = 250

        # --- Colors ---
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (30, 35, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_ACCENT = (100, 120, 255)
        self.NUCLEUS_COLORS = {
            "STABLE": (80, 100, 255),
            "EXCITED": (255, 220, 80),
            "CRITICAL": (255, 80, 80)
        }
        self.NEUTRON_DATA = {
            'STANDARD': {'color': (0, 180, 255), 'speed': 5, 'energy': 15, 'score': 100},
            'FAST': {'color': (0, 255, 150), 'speed': 8, 'energy': 8, 'score': 75},
            'HEAVY': {'color': (255, 100, 180), 'speed': 3.5, 'energy': 30, 'score': 250}
        }
        self.UNLOCK_THRESHOLDS = {
            'FAST': 1000,
            'HEAVY': 2500
        }

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 36, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.target_score = 0
        self.game_over = False
        self.launcher_angle = 0
        self.launch_cooldown = 0
        self.type_change_cooldown = 0
        self.unlocked_types = []
        self.selected_type_index = 0
        self.nucleus_pos = (0,0)
        self.nucleus_radius = 0
        self.nucleus_energy = 0.0
        self.nucleus_combo_timer = 0
        self.nucleus_combo_multiplier = 1
        self.projectiles = []
        self.particles = []
        self.last_space_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.target_score = self.INITIAL_TARGET_SCORE
        self.game_over = False

        self.launcher_angle = -math.pi / 2  # Straight up
        self.launch_cooldown = 0
        self.type_change_cooldown = 0

        self.unlocked_types = ['STANDARD']
        self.selected_type_index = 0

        self.nucleus_pos = (self.WIDTH // 2, self.HEIGHT // 2 - 20)
        self.nucleus_radius = 40
        self.nucleus_energy = 0.0
        self.nucleus_combo_timer = 0
        self.nucleus_combo_multiplier = 1

        self.projectiles = []
        self.particles = []
        self.last_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game State ---
        self._update_timers()
        self._update_nucleus()
        self._update_projectiles()
        self._update_particles()
        
        # --- Collision Detection & Resolution ---
        collision_reward, score_gain = self._check_collisions()
        reward += collision_reward
        self.score += score_gain

        # --- Check for Unlocks ---
        unlock_reward = self._check_unlocks()
        reward += unlock_reward

        # --- Finalize Step ---
        self.steps += 1
        terminated = self._check_termination()
        
        # --- Terminal Rewards ---
        if terminated:
            self.game_over = True
            if self.score >= self.target_score:
                reward += 100  # Win bonus
                # Optional: set up for next level
                # self.target_score += self.TARGET_SCORE_INCREMENT
            else:
                reward -= 10   # Loss penalty

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Aiming
        if movement == 1:  # Up
            self.launcher_angle -= 0.05
        elif movement == 2:  # Down
            self.launcher_angle += 0.05
        self.launcher_angle = max(-math.pi * 0.9, min(-math.pi * 0.1, self.launcher_angle))

        # Neutron Type Selection
        if self.type_change_cooldown == 0:
            if movement == 3:  # Left
                self.selected_type_index = (self.selected_type_index - 1) % len(self.unlocked_types)
                self.type_change_cooldown = 10
            elif movement == 4:  # Right
                self.selected_type_index = (self.selected_type_index + 1) % len(self.unlocked_types)
                self.type_change_cooldown = 10

        # Launching (edge-triggered)
        if space_held and not self.last_space_held and self.launch_cooldown == 0:
            self._launch_neutron()
            # No reward here, reward is tied to successful hits
        self.last_space_held = space_held

    def _update_timers(self):
        self.launch_cooldown = max(0, self.launch_cooldown - 1)
        self.type_change_cooldown = max(0, self.type_change_cooldown - 1)
        self.nucleus_combo_timer = max(0, self.nucleus_combo_timer - 1)

    def _update_nucleus(self):
        self.nucleus_energy = max(0, self.nucleus_energy - 0.05) # Natural decay
        if self.nucleus_combo_timer == 0:
            self.nucleus_combo_multiplier = 1

    def _update_projectiles(self):
        projectiles_to_keep = []
        for p in self.projectiles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            if 0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT:
                projectiles_to_keep.append(p)
        self.projectiles = projectiles_to_keep

    def _update_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # Tiny gravity effect
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                particles_to_keep.append(p)
        self.particles = particles_to_keep

    def _launch_neutron(self):
        # sfx: launch_sound.wav
        self.launch_cooldown = 20
        neutron_type_name = self.unlocked_types[self.selected_type_index]
        neutron_info = self.NEUTRON_DATA[neutron_type_name]
        
        start_pos = [self.WIDTH / 2, self.HEIGHT - 30]
        speed = neutron_info['speed']
        velocity = [speed * math.cos(self.launcher_angle), speed * math.sin(self.launcher_angle)]
        
        self.projectiles.append({
            'pos': start_pos,
            'vel': velocity,
            'type': neutron_type_name,
            'color': neutron_info['color'],
            'radius': 8
        })
        self._create_particles((start_pos[0], start_pos[1]), 10, neutron_info['color'], 0.5, self.launcher_angle, math.pi/8)

    def _check_collisions(self):
        total_reward = 0.0
        total_score = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            dist = math.hypot(p['pos'][0] - self.nucleus_pos[0], p['pos'][1] - self.nucleus_pos[1])
            if dist < self.nucleus_radius + p['radius']:
                # sfx: hit_sound.wav
                hit_reward, hit_score = self._process_hit(p)
                total_reward += hit_reward
                total_score += hit_score
            else:
                projectiles_to_keep.append(p)
        self.projectiles = projectiles_to_keep
        return total_reward, total_score

    def _process_hit(self, projectile):
        neutron_info = self.NEUTRON_DATA[projectile['type']]
        
        is_combo = self.nucleus_combo_timer > 0
        if is_combo:
            self.nucleus_combo_multiplier += 1
            # sfx: combo_up.wav
        else:
            self.nucleus_combo_multiplier = 1

        score_gain = int(neutron_info['score'] * self.nucleus_combo_multiplier)
        self.nucleus_energy += neutron_info['energy']
        self.nucleus_combo_timer = 90 # 1.5 seconds combo window

        reward = 1.0 # Base reward for a hit
        if is_combo:
            reward += 5.0 # Combo bonus reward
        
        self._create_particles(self.nucleus_pos, int(20 + self.nucleus_energy/2), projectile['color'], 2.0)
        return reward, score_gain

    def _check_unlocks(self):
        reward = 0
        for type_name, threshold in self.UNLOCK_THRESHOLDS.items():
            if type_name not in self.unlocked_types and self.score >= threshold:
                self.unlocked_types.append(type_name)
                reward += 10 # Unlock bonus reward
                # sfx: unlock_new_type.wav
        return reward

    def _check_termination(self):
        return self.steps >= self.MAX_STEPS or self.score >= self.target_score

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "target_score": self.target_score,
            "nucleus_energy": self.nucleus_energy,
            "combo_multiplier": self.nucleus_combo_multiplier,
            "unlocked_types": len(self.unlocked_types)
        }

    def _render_background(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)

    def _render_game(self):
        self._render_particles()
        self._render_nucleus()
        self._render_projectiles()
        self._render_launcher()

    def _render_nucleus(self):
        if self.nucleus_energy >= 70:
            color = self.NUCLEUS_COLORS["CRITICAL"]
        elif self.nucleus_energy >= 30:
            color = self.NUCLEUS_COLORS["EXCITED"]
        else:
            color = self.NUCLEUS_COLORS["STABLE"]
        
        # Pulsing effect based on energy
        pulse = (math.sin(self.steps * 0.1) * self.nucleus_energy / 20)
        radius = self.nucleus_radius + pulse
        
        self._draw_glowing_circle(self.screen, color, self.nucleus_pos, int(radius))

    def _render_launcher(self):
        launcher_pos = (self.WIDTH / 2, self.HEIGHT - 10)
        length = 30
        tip = (launcher_pos[0] + length * math.cos(self.launcher_angle),
               launcher_pos[1] + length * math.sin(self.launcher_angle))
        
        base_l = (launcher_pos[0] - 15 * math.sin(self.launcher_angle),
                  launcher_pos[1] + 15 * math.cos(self.launcher_angle))
        base_r = (launcher_pos[0] + 15 * math.sin(self.launcher_angle),
                  launcher_pos[1] - 15 * math.cos(self.launcher_angle))

        # Glowing base
        pygame.draw.circle(self.screen, self.COLOR_UI_ACCENT, (int(launcher_pos[0]), int(launcher_pos[1])), 20, 2)
        
        # Launcher body
        pygame.gfxdraw.aapolygon(self.screen, [(int(tip[0]), int(tip[1])), (int(base_l[0]), int(base_l[1])), (int(base_r[0]), int(base_r[1]))], self.COLOR_UI_TEXT)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(tip[0]), int(tip[1])), (int(base_l[0]), int(base_l[1])), (int(base_r[0]), int(base_r[1]))], self.COLOR_UI_TEXT)

    def _render_projectiles(self):
        for p in self.projectiles:
            self._draw_glowing_circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), p['radius'])

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = p['color'] + (alpha,)
            size = int(p['size'] * (p['lifespan'] / p['max_lifespan']))
            if size > 0:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

    def _render_ui(self):
        # Score and Target
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        target_text = self.font_ui.render(f"TARGET: {self.target_score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(target_text, (10, 30))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.metadata['render_fps']
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Combo Multiplier
        if self.nucleus_combo_multiplier > 1:
            combo_text = self.font_big.render(f"x{self.nucleus_combo_multiplier} COMBO!", True, self.NUCLEUS_COLORS["EXCITED"])
            text_rect = combo_text.get_rect(center=(self.WIDTH/2, 40))
            self.screen.blit(combo_text, text_rect)

        # Neutron Selector
        selector_y = self.HEIGHT - 25
        for i, type_name in enumerate(self.unlocked_types):
            color = self.NEUTRON_DATA[type_name]['color']
            x_pos = self.WIDTH/2 + (i - (len(self.unlocked_types)-1)/2) * 40
            radius = 12 if i == self.selected_type_index else 8
            self._draw_glowing_circle(self.screen, color, (int(x_pos), selector_y), radius)
            if i == self.selected_type_index:
                pygame.draw.circle(self.screen, (255,255,255), (int(x_pos), selector_y), radius+3, 2)


    def _create_particles(self, pos, count, color, speed_multiplier, angle_base=None, angle_spread=math.pi*2):
        for _ in range(count):
            if angle_base is not None:
                angle = angle_base + random.uniform(-angle_spread/2, angle_spread/2)
            else:
                angle = random.uniform(0, 2 * math.pi)

            speed = random.uniform(1, 3) * speed_multiplier
            lifespan = random.randint(20, 50)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color,
                'size': random.uniform(2, 5)
            })

    def _draw_glowing_circle(self, surface, color, center, radius):
        # Draw glow
        glow_radius = int(radius * 1.8)
        glow_alpha = 60
        temp_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, color + (glow_alpha,), (glow_radius, glow_radius), glow_radius)
        surface.blit(temp_surf, (center[0] - glow_radius, center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Draw main circle
        pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), int(radius), color)
        pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius), color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run with display, comment out the os.environ line at the top
    # and instantiate GameEnv with render_mode="human"
    
    # Check if we can use display
    render_mode = "rgb_array"
    try:
        pygame.display.init()
        pygame.display.set_mode((1,1))
        pygame.display.quit()
        # If the above works, we can probably use a display
        if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
             print("Running in headless mode. No display will be shown.")
        else:
             render_mode = "human"
    except pygame.error:
        print("Pygame display error. Running in headless mode.")


    env = GameEnv(render_mode=render_mode)
    obs, info = env.reset()
    
    if render_mode == "human":
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Nuclear Chain Reaction")
    
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n--- Manual Controls ---")
    print("W/S or Up/Down Arrow: Aim")
    print("A/D or Left/Right Arrow: Select Neutron Type")
    print("Spacebar: Launch Neutron")
    print("R: Reset Environment")
    print("-----------------------\n")

    while not done:
        # Action mapping for human play
        movement = 0 # none
        space_held = 0
        shift_held = 0

        # This part only works if a display is available
        if render_mode == "human":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space_held = 1
            
            if keys[pygame.K_r]:
                obs, info = env.reset()
                total_reward = 0
                print("Environment Reset!")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
        else: # In headless mode, just take random actions
            action = env.action_space.sample()

        if render_mode == "human":
            action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            if render_mode != "human": # stop after one episode in headless
                done = True

        # Render the observation from the environment
        if render_mode == "human":
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        clock.tick(env.metadata['render_fps'])

    env.close()