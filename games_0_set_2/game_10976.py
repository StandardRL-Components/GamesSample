import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:14:47.866029
# Source Brief: brief_00976.md
# Brief Index: 976
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: A real-time puzzle game where the player activates one of two magnets
    to group 10 metallic cubes.

    - **Goal**: Group all 10 cubes within a single magnet's attraction radius.
    - **Mechanics**:
        - Activating a magnet pulls all cubes towards it.
        - When a cube enters a magnet's radius, the *opposite* magnet's strength increases.
    - **Challenge**: The increasing strength of the magnets creates chaotic,
      unpredictable cube movements, requiring strategic timing.
    - **Action Space**: MultiDiscrete([5, 2, 2])
        - action[0]: Movement (unused)
        - action[1]: Activate Red Magnet (space bar)
        - action[2]: Activate Blue Magnet (shift key)
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Activate one of two magnets to group 10 metallic cubes. "
        "Using a magnet increases the strength of its opposite, creating chaotic and challenging physics."
    )
    user_guide = (
        "Press space to activate the red magnet and shift to activate the blue magnet. "
        "Group all cubes in one magnet's field to win."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    TIME_LIMIT_SECONDS = 45
    TIME_LIMIT_STEPS = TIME_LIMIT_SECONDS * FPS
    MAX_EPISODE_STEPS = 3000

    # Colors
    COLOR_BG = (20, 25, 35)
    COLOR_RED_MAGNET = (255, 80, 80)
    COLOR_BLUE_MAGNET = (80, 150, 255)
    COLOR_CUBE = (190, 195, 205)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (10, 10, 15)

    # Game Parameters
    NUM_CUBES = 10
    CUBE_SIZE = 10
    CUBE_DRAG = 0.96
    MAGNET_BASE_STRENGTH = 1.0
    MAGNET_FORCE_MULTIPLIER = 0.008
    MAGNET_BASE_RADIUS = 30
    MAGNET_ATTRACTION_RADIUS_SCALE = 80
    MAGNET_STRENGTH_INCREASE = 1.05

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_timer = pygame.font.SysFont("monospace", 28, bold=True)

        self.magnets = []
        self.cubes = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.active_magnet_idx = None
        self.cubes_in_radius_last_step = {0: set(), 1: set()}
        
        # This call is not strictly necessary but good practice for dev
        if self.reset() is None:
             raise Exception("Reset method failed to initialize properly.")

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT_STEPS
        self.active_magnet_idx = None

        self.magnets = [
            {
                "pos": pygame.Vector2(self.WIDTH * 0.2, self.HEIGHT / 2),
                "color": self.COLOR_RED_MAGNET,
                "strength": self.MAGNET_BASE_STRENGTH,
            },
            {
                "pos": pygame.Vector2(self.WIDTH * 0.8, self.HEIGHT / 2),
                "color": self.COLOR_BLUE_MAGNET,
                "strength": self.MAGNET_BASE_STRENGTH,
            },
        ]

        self.cubes = []
        while len(self.cubes) < self.NUM_CUBES:
            pos = pygame.Vector2(
                self.np_random.uniform(self.CUBE_SIZE, self.WIDTH - self.CUBE_SIZE),
                self.np_random.uniform(self.CUBE_SIZE, self.HEIGHT - self.CUBE_SIZE),
            )
            # Ensure cubes don't spawn inside a magnet's initial radius
            too_close = False
            for magnet in self.magnets:
                if pos.distance_to(magnet["pos"]) < self.MAGNET_BASE_RADIUS * 2:
                    too_close = True
                    break
            if not too_close:
                self.cubes.append({"pos": pos, "vel": pygame.Vector2(0, 0)})
        
        self.cubes_in_radius_last_step = {0: set(), 1: set()}

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1
        reward = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self.active_magnet_idx = None
        if space_held and not shift_held:
            self.active_magnet_idx = 0  # Red magnet
            # SFX: ZAP_RED.WAV
        elif shift_held and not space_held:
            self.active_magnet_idx = 1  # Blue magnet
            # SFX: ZAP_BLUE.WAV

        # --- Physics and Game Logic ---
        if self.active_magnet_idx is not None:
            active_magnet = self.magnets[self.active_magnet_idx]
            for cube in self.cubes:
                direction = active_magnet["pos"] - cube["pos"]
                dist_sq = direction.length_squared()
                if dist_sq > 1:
                    force = direction.normalize() * (active_magnet["strength"] * self.MAGNET_FORCE_MULTIPLIER)
                    cube["vel"] += force

        for cube in self.cubes:
            cube["vel"] *= self.CUBE_DRAG
            cube["pos"] += cube["vel"]

            # Boundary checks
            if cube["pos"].x < self.CUBE_SIZE / 2:
                cube["pos"].x = self.CUBE_SIZE / 2
                cube["vel"].x *= -0.8
            elif cube["pos"].x > self.WIDTH - self.CUBE_SIZE / 2:
                cube["pos"].x = self.WIDTH - self.CUBE_SIZE / 2
                cube["vel"].x *= -0.8
            if cube["pos"].y < self.CUBE_SIZE / 2:
                cube["pos"].y = self.CUBE_SIZE / 2
                cube["vel"].y *= -0.8
            elif cube["pos"].y > self.HEIGHT - self.CUBE_SIZE / 2:
                cube["pos"].y = self.HEIGHT - self.CUBE_SIZE / 2
                cube["vel"].y *= -0.8

        # --- Magnet Strength and Reward Logic ---
        cubes_in_radius_this_step = {0: set(), 1: set()}
        counts = [0, 0]
        for i, cube in enumerate(self.cubes):
            for mag_idx, magnet in enumerate(self.magnets):
                attraction_radius = self.MAGNET_BASE_RADIUS + self._get_attraction_radius(magnet["strength"])
                if cube["pos"].distance_to(magnet["pos"]) < attraction_radius:
                    cubes_in_radius_this_step[mag_idx].add(i)
                    counts[mag_idx] += 1
        
        for i in range(self.NUM_CUBES):
            if i in cubes_in_radius_this_step[0]: # In red radius
                self.magnets[1]["strength"] *= self.MAGNET_STRENGTH_INCREASE
            if i in cubes_in_radius_this_step[1]: # In blue radius
                self.magnets[0]["strength"] *= self.MAGNET_STRENGTH_INCREASE

        # Continuous reward
        reward += (counts[0] + counts[1]) * 0.1

        # Event-based reward
        newly_in_red = cubes_in_radius_this_step[0] - self.cubes_in_radius_last_step[0]
        newly_in_blue = cubes_in_radius_this_step[1] - self.cubes_in_radius_last_step[1]
        reward += (len(newly_in_red) + len(newly_in_blue)) * 5.0
        if len(newly_in_red) > 0 or len(newly_in_blue) > 0:
            # SFX: CUBE_CAPTURE.WAV
            pass

        self.cubes_in_radius_last_step = cubes_in_radius_this_step

        # --- Termination Check ---
        terminated = False
        if counts[0] == self.NUM_CUBES or counts[1] == self.NUM_CUBES:
            terminated = True
            reward += 100
            # SFX: VICTORY.WAV
        elif self.time_remaining <= 0 or self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            reward -= 100
            # SFX: FAILURE.WAV
        
        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_attraction_radius(self, strength):
        return min(math.log(max(1, strength)) * self.MAGNET_ATTRACTION_RADIUS_SCALE, self.WIDTH)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw attraction fields first
        for i, magnet in enumerate(self.magnets):
            attraction_radius = self.MAGNET_BASE_RADIUS + self._get_attraction_radius(magnet["strength"])
            color = list(magnet["color"])
            
            # Draw a faint, static representation of the full attraction field
            surface = self.screen.copy()
            pygame.gfxdraw.filled_circle(surface, int(magnet["pos"].x), int(magnet["pos"].y), int(attraction_radius), (*color, 20))
            pygame.gfxdraw.aacircle(surface, int(magnet["pos"].x), int(magnet["pos"].y), int(attraction_radius), (*color, 40))
            self.screen.blit(surface, (0,0), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw pulsing field for active magnet
        if self.active_magnet_idx is not None:
            active_magnet = self.magnets[self.active_magnet_idx]
            pulse = (math.sin(self.steps * 0.4) + 1) / 2  # 0 to 1
            radius = self.MAGNET_BASE_RADIUS + self._get_attraction_radius(active_magnet["strength"])
            pulse_radius = int(radius * (0.8 + pulse * 0.2))
            alpha = int(80 + pulse * 40)
            pygame.gfxdraw.aacircle(self.screen, int(active_magnet["pos"].x), int(active_magnet["pos"].y), pulse_radius, (*active_magnet["color"], alpha))
            pygame.gfxdraw.aacircle(self.screen, int(active_magnet["pos"].x), int(active_magnet["pos"].y), pulse_radius-1, (*active_magnet["color"], alpha//2))


        # Draw cubes and their glows
        for i, cube in enumerate(self.cubes):
            glow_color = None
            if i in self.cubes_in_radius_last_step[0]:
                glow_color = self.magnets[0]["color"]
            elif i in self.cubes_in_radius_last_step[1]:
                glow_color = self.magnets[1]["color"]
            
            cube_rect = pygame.Rect(
                cube["pos"].x - self.CUBE_SIZE / 2,
                cube["pos"].y - self.CUBE_SIZE / 2,
                self.CUBE_SIZE,
                self.CUBE_SIZE,
            )

            if glow_color:
                glow_surface = pygame.Surface((self.CUBE_SIZE*2, self.CUBE_SIZE*2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (*glow_color, 60), (self.CUBE_SIZE, self.CUBE_SIZE), self.CUBE_SIZE)
                pygame.draw.circle(glow_surface, (*glow_color, 30), (self.CUBE_SIZE, self.CUBE_SIZE), self.CUBE_SIZE*0.7)
                self.screen.blit(glow_surface, (cube_rect.centerx - self.CUBE_SIZE, cube_rect.centery - self.CUBE_SIZE), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.draw.rect(self.screen, self.COLOR_CUBE, cube_rect, border_radius=2)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.COLOR_CUBE), cube_rect, width=1, border_radius=2)

        # Draw magnets on top
        for magnet in self.magnets:
            visual_radius = int(self.MAGNET_BASE_RADIUS * (1 + math.log10(magnet["strength"])))
            pos = (int(magnet["pos"].x), int(magnet["pos"].y))
            
            # Draw a subtle darker base for depth
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], visual_radius, tuple(c*0.5 for c in magnet["color"]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], visual_radius, tuple(c*0.5 for c in magnet["color"]))
            
            # Draw the main magnet body with a highlight
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], visual_radius - 2, magnet["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], visual_radius - 2, magnet["color"])
            
            # Highlight effect
            highlight_pos = (pos[0] - visual_radius//3, pos[1] - visual_radius//3)
            highlight_rad = visual_radius//2
            pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], highlight_rad, (255, 255, 255, 40))


    def _render_ui(self):
        def draw_text(text, pos, font, color, shadow_color, shadow_offset=(1, 1)):
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Score
        score_text = f"SCORE: {int(self.score)}"
        draw_text(score_text, (10, 10), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Timer
        time_left_sec = max(0, self.time_remaining / self.FPS)
        timer_text = f"{time_left_sec:.1f}"
        timer_color = self.COLOR_TEXT if time_left_sec > 10 else self.COLOR_RED_MAGNET
        text_width = self.font_timer.size(timer_text)[0]
        draw_text(timer_text, (self.WIDTH - text_width - 15, 10), self.font_timer, timer_color, self.COLOR_TEXT_SHADOW, (2, 2))

        # Magnet Info
        for i, magnet in enumerate(self.magnets):
            pos = magnet["pos"]
            strength_text = f"STR: {magnet['strength']:.1f}x"
            count_text = f"CUBES: {len(self.cubes_in_radius_last_step[i])}/{self.NUM_CUBES}"
            
            draw_text(strength_text, (pos.x - 40, pos.y + 40), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
            draw_text(count_text, (pos.x - 40, pos.y + 60), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "red_magnet_strength": self.magnets[0]["strength"],
            "blue_magnet_strength": self.magnets[1]["strength"],
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        # print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Magnet Mania")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # 0=none
        space_held = 0 # 0=released
        shift_held = 0 # 0=released

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESET ---")
                obs, info = env.reset()
                total_reward = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # The env automatically handles game over, but for manual play we can wait for a reset
            
        # --- Rendering ---
        # The observation is already the rendered image
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()