import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:43:58.170615
# Source Brief: brief_00664.md
# Brief Index: 664
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stabilize a cosmic energy channel by countering disruptive waves. "
        "Deploy nanobots and activate terraforming fields to maximize energy flow."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. "
        "Press space to deploy a nanobot and shift to activate a terraforming field."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_CHANNEL = (50, 80, 150, 100) # RGBA
    COLOR_WAVE = (255, 50, 50)
    COLOR_NANOBOT = (50, 255, 150)
    COLOR_TERRAFORM = (180, 50, 255)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BAR = (40, 200, 220)
    COLOR_UI_BAR_BG = (40, 40, 80)

    # Game Parameters
    CURSOR_SPEED = 12
    DEPLOY_COOLDOWN = 10 # frames
    TERRAFORM_COOLDOWN = 15 # frames
    TERRAFORM_COST = 20
    TERRAFORM_RADIUS = 30
    TERRAFORM_STRENGTH = 0.5 # Damping factor on waves
    NANOBOT_RADIUS = 5
    NANOBOT_CALM_STRENGTH = 0.015 # Each bot reduces total disruption
    BASE_CHANNEL_POWER = 20.0 # Energy per step with no waves
    DISRUPTION_ENERGY_SCALAR = 0.005 # How much disruption reduces energy
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)
        
        # Initialize state variables
        self.cursor_pos = None
        self.stars = None
        self.waves = None
        self.nanobots = None
        self.terraforms = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.available_nanobots = None
        self.terraform_energy = None
        self.deploy_cooldown_timer = None
        self.terraform_cooldown_timer = None
        self.wave_base_amplitude = None
        self.wave_base_frequency = None
        self.nanobot_type = None
        self.terraform_level = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        
        self.waves = []
        self.nanobots = []
        self.terraforms = []
        self.particles = []

        self.available_nanobots = 10
        self.terraform_energy = 100.0
        
        self.deploy_cooldown_timer = 0
        self.terraform_cooldown_timer = 0
        
        self.wave_base_amplitude = 20.0
        self.wave_base_frequency = 0.02
        
        self.nanobot_type = 1
        self.terraform_level = 1

        self._create_stars(200)
        for _ in range(3):
            self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)
        
        # --- Game Logic Update ---
        self._update_state()
        self._update_waves()
        self._update_nanobots()
        self._update_particles()

        # --- Calculate Reward & Score ---
        reward, energy_this_step = self._calculate_reward_and_score()
        self.score += energy_this_step
        
        # --- Check for Termination ---
        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.score > 7500: # Skilled play target
                reward += 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 100, self.HEIGHT - 100) # Confine to channel

        # --- Nanobot Deployment ---
        self.action_deployed_bot = False
        if space_held and self.deploy_cooldown_timer == 0 and self.available_nanobots > 0:
            self.nanobots.append({
                "pos": self.cursor_pos.copy(),
                "type": self.nanobot_type,
                "anim_offset": self.np_random.random() * 2 * math.pi
            })
            self.available_nanobots -= 1
            self.deploy_cooldown_timer = self.DEPLOY_COOLDOWN
            self.action_deployed_bot = True
            # sfx: nanobot_deploy.wav
            self._create_particles(self.cursor_pos, self.COLOR_NANOBOT, 20, 3)

        # --- Terraforming ---
        self.action_terraformed = False
        if shift_held and self.terraform_cooldown_timer == 0 and self.terraform_energy >= self.TERRAFORM_COST:
            self.terraforms.append({
                "pos": self.cursor_pos.copy(),
                "radius": self.TERRAFORM_RADIUS * self.terraform_level,
                "strength": self.TERRAFORM_STRENGTH
            })
            self.terraform_energy -= self.TERRAFORM_COST
            self.terraform_cooldown_timer = self.TERRAFORM_COOLDOWN
            self.action_terraformed = True
            # sfx: terraform_activate.wav
            self._create_particles(self.cursor_pos, self.COLOR_TERRAFORM, 30, 2)

    def _update_state(self):
        # Update cooldowns
        if self.deploy_cooldown_timer > 0: self.deploy_cooldown_timer -= 1
        if self.terraform_cooldown_timer > 0: self.terraform_cooldown_timer -= 1
        
        # Regenerate resources
        self.terraform_energy = min(100.0, self.terraform_energy + 0.1)
        if self.steps % 60 == 0:
            self.available_nanobots = min(20, self.available_nanobots + 1)
        
        # Spawn new waves
        if self.steps % 75 == 0:
            self._spawn_wave()
        
        # Difficulty progression
        if self.steps > 0 and self.steps % 200 == 0:
            self.wave_base_amplitude *= 1.15
            self.wave_base_frequency *= 1.1
        
        # Unlocks
        if self.steps == 300: self.nanobot_type = 2
        if self.steps == 600: self.nanobot_type = 3
        if self.steps == 400: self.terraform_level = 1.2
        if self.steps == 700: self.terraform_level = 1.5

    def _spawn_wave(self):
        y_offset = self.HEIGHT // 2 + self.np_random.uniform(-50, 50)
        amplitude = self.wave_base_amplitude * self.np_random.uniform(0.8, 1.2)
        frequency = self.wave_base_frequency * self.np_random.uniform(0.8, 1.2)
        phase = self.np_random.random() * 2 * math.pi
        speed = self.np_random.uniform(1.5, 2.5)
        self.waves.append({
            "y_offset": y_offset, "amp": amplitude, "freq": frequency,
            "phase": phase, "speed": speed
        })

    def _update_waves(self):
        for wave in self.waves:
            wave["phase"] += wave["speed"] * 0.1

    def _update_nanobots(self):
        for bot in self.nanobots:
            bot["pos"].y = self._get_wave_y(bot["pos"].x)
            bot["anim_offset"] += 0.2

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _calculate_reward_and_score(self):
        total_disruption = 0
        for x in range(0, self.WIDTH, 10):
            wave_y = self._get_wave_y(x, apply_terraform=True)
            base_y = self.HEIGHT // 2
            total_disruption += abs(wave_y - base_y)
        
        nanobot_calming_factor = 1.0 - (len(self.nanobots) * self.NANOBOT_CALM_STRENGTH * self.nanobot_type)
        final_disruption = max(0, total_disruption * nanobot_calming_factor)
        
        energy_this_step = max(0, self.BASE_CHANNEL_POWER - final_disruption * self.DISRUPTION_ENERGY_SCALAR)
        
        reward = energy_this_step * 0.1
        if self.action_deployed_bot: reward += 1.0
        if self.action_terraformed: reward += 2.0
        
        return reward, energy_this_step

    def _get_wave_y(self, x, apply_terraform=False):
        total_y = 0
        for wave in self.waves:
            total_y += math.sin(x * wave["freq"] + wave["phase"]) * wave["amp"]

        y = self.HEIGHT // 2 + total_y
        
        if apply_terraform:
            for terra in self.terraforms:
                dist = terra["pos"].distance_to(pygame.Vector2(x, y))
                if dist < terra["radius"]:
                    damping = 1.0 - (1.0 - dist / terra["radius"]) * terra["strength"]
                    y = self.HEIGHT // 2 + (y - self.HEIGHT // 2) * damping
        return y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_channel()
        self._render_terraforms()
        self._render_waves()
        self._render_nanobots()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "available_nanobots": self.available_nanobots,
            "terraform_energy": self.terraform_energy,
        }

    # --- Rendering Methods ---

    def _render_stars(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], (int(star['x']), int(star['y'])), star['size'])

    def _render_channel(self):
        channel_rect = pygame.Rect(0, 100, self.WIDTH, self.HEIGHT - 200)
        s = pygame.Surface((self.WIDTH, self.HEIGHT-200), pygame.SRCALPHA)
        s.fill(self.COLOR_CHANNEL)
        self.screen.blit(s, (0, 100))

    def _render_terraforms(self):
        for terra in self.terraforms:
            # Draw a subtle visual effect for the terraformed area
            s = pygame.Surface((terra["radius"] * 2, terra["radius"] * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_TERRAFORM + (50,), (terra["radius"], terra["radius"]), terra["radius"])
            self.screen.blit(s, (terra["pos"].x - terra["radius"], terra["pos"].y - terra["radius"]), special_flags=pygame.BLEND_RGBA_ADD)


    def _render_waves(self):
        for wave in self.waves:
            points = []
            for x in range(0, self.WIDTH + 1, 5):
                y = self._get_wave_y(x)
                points.append((x, y))
            if len(points) > 1:
                pygame.draw.aalines(self.screen, self.COLOR_WAVE, False, points, 1)

    def _render_nanobots(self):
        for bot in self.nanobots:
            pos = (int(bot["pos"].x), int(bot["pos"].y))
            # Glow effect
            glow_radius = int(self.NANOBOT_RADIUS * (2.0 + 0.5 * math.sin(bot["anim_offset"])))
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_NANOBOT + (30,), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
            # Core
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.NANOBOT_RADIUS, self.COLOR_NANOBOT)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.NANOBOT_RADIUS, self.COLOR_NANOBOT)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, int(255 * (p["life"] / p["max_life"])))
            color = p["color"] + (alpha,)
            size = int(p["size"] * (p["life"] / p["max_life"]))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size,size), size)
                self.screen.blit(s, (p["pos"].x - size, p["pos"].y - size), special_flags=pygame.BLEND_RGBA_ADD)


    def _render_cursor(self):
        pos = (int(self.cursor_pos.x), int(self.cursor_pos.y))
        size = 12
        # Glow
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size + 2, self.COLOR_CURSOR)
        # Crosshair
        pygame.draw.aaline(self.screen, self.COLOR_CURSOR, (pos[0] - size, pos[1]), (pos[0] + size, pos[1]))
        pygame.draw.aaline(self.screen, self.COLOR_CURSOR, (pos[0], pos[1] - size), (pos[0], pos[1] + size))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"ENERGY: {int(self.score):,}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Steps
        steps_text = self.font_main.render(f"CYCLE: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        # Nanobots
        bot_text = self.font_main.render(f"NANOBOTS: {self.available_nanobots}", True, self.COLOR_UI_TEXT)
        self.screen.blit(bot_text, (10, self.HEIGHT - 30))
        
        # Terraforming Energy
        bar_width = 150
        bar_height = 20
        bar_x = self.WIDTH - bar_width - 10
        bar_y = self.HEIGHT - bar_height - 10
        
        fill_ratio = self.terraform_energy / 100.0
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y, int(bar_width * fill_ratio), bar_height))
        
        terraform_text = self.font_small.render("TERRAFORM ENERGY", True, self.COLOR_UI_TEXT)
        self.screen.blit(terraform_text, (bar_x, bar_y - 18))

    # --- Utility Methods ---

    def _create_stars(self, num_stars):
        self.stars = []
        for _ in range(num_stars):
            self.stars.append({
                'x': self.np_random.integers(0, self.WIDTH),
                'y': self.np_random.integers(0, self.HEIGHT),
                'size': self.np_random.integers(1, 3),
                'color': (
                    self.np_random.integers(100, 200),
                    self.np_random.integers(100, 200),
                    self.np_random.integers(150, 255)
                )
            })

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * max_speed
            life = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "color": color,
                "life": life,
                "max_life": life,
                "size": self.np_random.integers(2, 5)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This function is for internal validation and can be removed in production
        print("Running internal validation...")
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
    # This block allows you to play the game manually
    # It requires a graphical display.
    os.environ["SDL_VIDEODRIVER"] = "" # Unset dummy driver for manual play
    pygame.display.init()
    pygame.display.set_caption("GameEnv Manual Control")
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a window for manual play
    render_window = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    running = True
    terminated = False
    
    # Store action states
    movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space_held = 0 # 0=released, 1=held
    shift_held = 0 # 0=released, 1=held

    print("\n--- Manual Control ---")
    print("Arrows: Move cursor")
    print("Space: Deploy nanobot")
    print("Shift: Terraform")
    print("Q: Quit")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Use a separate Pygame window for rendering the observation
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_window.blit(render_surface, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}")
            obs, info = env.reset()

    env.close()