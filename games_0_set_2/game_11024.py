import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:19:10.732506
# Source Brief: brief_01024.md
# Brief Index: 1024
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
        "Infiltrate a high-tech facility as a cybernetic ghost. Use camouflage to evade colored spotlights "
        "and terraform chasms to reach the data core."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to cycle camouflage color. "
        "Press shift to create a platform over a chasm."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 40
    GRID_W = SCREEN_WIDTH // GRID_SIZE
    GRID_H = SCREEN_HEIGHT // GRID_SIZE
    MAX_STEPS = 2000
    FPS = 30

    # --- COLORS ---
    COLOR_BG = (10, 5, 25)
    COLOR_OBSTACLE = (0, 0, 0)
    COLOR_GRID_HINT = (20, 15, 45)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_CORE = (255, 255, 100)
    COLOR_PLATFORM = (50, 200, 255)
    
    CAMO_COLORS = [
        (255, 50, 50),   # 0: Red
        (50, 255, 50),   # 1: Green
        (50, 50, 255),   # 2: Blue
        (200, 200, 200)  # 3: Off (White)
    ]
    
    CAMO_OFF_IDX = 3
    MAX_TERRAFORM = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- GYM SPACES ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- PYGAME SETUP ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_big = pygame.font.Font(None, 50)

        # --- STATE VARIABLES (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0, 0])
        self.player_render_pos = np.array([0.0, 0.0])
        self.player_camo_idx = self.CAMO_OFF_IDX
        self.terraform_uses = 0
        self.obstacles = set()
        self.terraformed_platforms = set()
        self.data_core_pos = np.array([0, 0])
        self.spotlights = []
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False

        # --- STATIC BACKGROUND ELEMENTS ---
        self._background_stars = [(random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT), random.randint(1, 2)) for _ in range(100)]
        self._background_buildings = self._generate_background_buildings()

    def _generate_background_buildings(self):
        buildings = []
        for i in range(20):
            w = random.randint(30, 100)
            h = random.randint(50, 250)
            x = random.randint(-w, self.SCREEN_WIDTH)
            color_val = random.randint(15, 35)
            color = (color_val, color_val, color_val + 20)
            buildings.append(pygame.Rect(x, self.SCREEN_HEIGHT - h, w, h))
            buildings.append(color)
        return buildings

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([1, self.GRID_H // 2])
        self.player_render_pos = self.player_pos.astype(float) * self.GRID_SIZE + self.GRID_SIZE / 2
        
        self.player_camo_idx = self.CAMO_OFF_IDX
        self.terraform_uses = self.MAX_TERRAFORM
        
        self.data_core_pos = np.array([self.GRID_W - 2, self.GRID_H // 2])
        
        self._generate_level()
        self._generate_spotlights()

        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.obstacles = set()
        self.terraformed_platforms = set()
        # Create vertical chasms
        for chasm_x in [self.GRID_W // 3, 2 * self.GRID_W // 3]:
            for y in range(self.GRID_H):
                self.obstacles.add((chasm_x, y))
        # Remove a few blocks to create paths
        for _ in range(self.GRID_H // 3):
            self.obstacles.discard((self.GRID_W // 3, self.np_random.integers(0, self.GRID_H)))
            self.obstacles.discard((2 * self.GRID_W // 3, self.np_random.integers(0, self.GRID_H)))
        # Ensure start and end are not obstacles
        self.obstacles.discard(tuple(self.player_pos))
        self.obstacles.discard(tuple(self.data_core_pos))

    def _generate_spotlights(self):
        self.spotlights = []
        # Pattern 1: Vertical patrol
        self.spotlights.append({
            "pos": np.array([self.GRID_W / 4 * self.GRID_SIZE, 0.0]),
            "start": np.array([self.GRID_W / 4 * self.GRID_SIZE, 0.0]),
            "end": np.array([self.GRID_W / 4 * self.GRID_SIZE, self.SCREEN_HEIGHT]),
            "color_idx": 0, "radius": 70, "t": 0.0, "speed": 0.01
        })
        # Pattern 2: Horizontal patrol
        self.spotlights.append({
            "pos": np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2]),
            "start": np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2]),
            "end": np.array([self.SCREEN_WIDTH, self.SCREEN_HEIGHT / 2]),
            "color_idx": 1, "radius": 90, "t": 0.0, "speed": 0.007
        })
        # Pattern 3: Circular patrol
        self.spotlights.append({
            "center": np.array([self.GRID_W * 0.8 * self.GRID_SIZE, self.SCREEN_HEIGHT / 2]),
            "orbit_radius": 100, "angle": 0.0,
            "color_idx": 2, "radius": 60, "speed": 0.02
        })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1
        
        # --- HANDLE ACTIONS ---
        # Action: Terraform (on press)
        if shift_held and not self.prev_shift_held and self.terraform_uses > 0:
            player_pos_tuple = tuple(self.player_pos)
            if player_pos_tuple in self.obstacles:
                self.obstacles.remove(player_pos_tuple)
                self.terraformed_platforms.add(player_pos_tuple)
                reward += 5.0 # Rewarded for useful terraforming
                # Sound: Terraform_Create
                self._create_particles(self.player_render_pos, self.COLOR_PLATFORM, 20)
            else:
                reward -= 1.0 # Penalized for useless terraforming
                # Sound: Terraform_Fail
            self.terraform_uses -= 1

        # Action: Cycle Camo (on press)
        if space_held and not self.prev_space_held:
            self.player_camo_idx = (self.player_camo_idx + 1) % len(self.CAMO_COLORS)
            # Sound: Camo_Switch

        # Action: Movement
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            next_pos = self.player_pos + np.array([dx, dy])
            if (0 <= next_pos[0] < self.GRID_W and
                0 <= next_pos[1] < self.GRID_H and
                tuple(next_pos) not in self.obstacles):
                self.player_pos = next_pos

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- UPDATE GAME STATE ---
        self._update_spotlights()
        self._update_particles()
        
        # --- CHECK TERMINATION & REWARDS ---
        terminated = False
        
        # 1. Detection by spotlights
        player_center = self.player_pos * self.GRID_SIZE + self.GRID_SIZE / 2
        for s in self.spotlights:
            dist = np.linalg.norm(player_center - s.get("pos", s.get("center")))
            if dist < s["radius"] and self.player_camo_idx != s["color_idx"]:
                terminated = True
                reward = -100.0
                self.game_over = True
                # Sound: Detection_Alarm
                break
        
        if not terminated:
            # 2. Reached data core
            if np.array_equal(self.player_pos, self.data_core_pos):
                terminated = True
                reward = 100.0
                self.score += 1 # A different kind of score, counting wins
                self.game_over = True
                # Sound: Win_Jingle
            # 3. Max steps
            elif self.steps >= self.MAX_STEPS:
                terminated = True
                # No specific reward change, just ends.
            else:
                # 4. Survival reward
                reward += 0.1

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_spotlights(self):
        base_speed_multiplier = 1.0 + (self.steps / 1000) * 0.05
        for s in self.spotlights:
            if "t" in s: # Linear patrol
                s["t"] += s["speed"] * base_speed_multiplier
                if s["t"] > 1.0 or s["t"] < 0.0:
                    s["speed"] *= -1
                    s["t"] = np.clip(s["t"], 0.0, 1.0)
                s["pos"] = s["start"] + (s["end"] - s["start"]) * s["t"]
            elif "angle" in s: # Circular patrol
                s["angle"] += s["speed"] * base_speed_multiplier
                s["pos"] = s["center"] + np.array([math.cos(s["angle"]), math.sin(s["angle"])]) * s["orbit_radius"]

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy().astype(float),
                "vel": velocity,
                "life": random.randint(15, 30),
                "color": color
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["life"] -= 1

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "terraform_uses": self.terraform_uses,
            "player_pos": tuple(self.player_pos),
        }

    def _render_all(self):
        # --- Smooth player movement interpolation ---
        target_render_pos = self.player_pos.astype(float) * self.GRID_SIZE + self.GRID_SIZE / 2
        self.player_render_pos += (target_render_pos - self.player_render_pos) * 0.5

        # --- DRAWING ---
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_level()
        self._render_spotlights()
        self._render_particles()
        self._render_player()
        self._render_ui()

        if self.game_over:
            self._render_game_over()

    def _render_background(self):
        for x, y, size in self._background_stars:
            pygame.draw.rect(self.screen, (200, 200, 255), (x, y, size, size))
        
        for i in range(0, len(self._background_buildings), 2):
            rect = self._background_buildings[i]
            color = self._background_buildings[i+1]
            pygame.draw.rect(self.screen, color, rect)

    def _render_level(self):
        # Render grid hints
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID_HINT, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID_HINT, (0, y), (self.SCREEN_WIDTH, y))

        # Render obstacles (chasms)
        for x, y in self.obstacles:
            rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
        
        # Render terraformed platforms
        for x, y in self.terraformed_platforms:
            rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, rect)
            pygame.draw.rect(self.screen, (255,255,255), rect, 1)

        # Render data core
        core_pos_px = self.data_core_pos * self.GRID_SIZE + self.GRID_SIZE / 2
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        radius = int(self.GRID_SIZE * 0.4 + pulse * 5)
        alpha = int(150 + pulse * 105)
        color = self.COLOR_CORE
        
        s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, color + (alpha,), (radius, radius), radius)
        self.screen.blit(s, (int(core_pos_px[0] - radius), int(core_pos_px[1] - radius)))
        pygame.gfxdraw.aacircle(self.screen, int(core_pos_px[0]), int(core_pos_px[1]), radius, color)

    def _render_spotlights(self):
        for s in self.spotlights:
            pos = s.get("pos", s.get("center"))
            radius = int(s["radius"])
            color = self.CAMO_COLORS[s["color_idx"]]
            
            # Create a transparent surface for the spotlight
            spot_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(spot_surface, color + (40,), (radius, radius), radius)
            
            self.screen.blit(spot_surface, (int(pos[0] - radius), int(pos[1] - radius)))
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, color + (100,))

    def _render_player(self):
        pos_px = self.player_render_pos
        size = self.GRID_SIZE * 0.7
        
        # Draw camo aura
        aura_color = self.CAMO_COLORS[self.player_camo_idx]
        if self.player_camo_idx != self.CAMO_OFF_IDX:
            aura_radius = int(self.GRID_SIZE * 0.6)
            aura_surface = pygame.Surface((aura_radius*2, aura_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(aura_surface, aura_color + (100,), (aura_radius, aura_radius), aura_radius)
            self.screen.blit(aura_surface, (int(pos_px[0]-aura_radius), int(pos_px[1]-aura_radius)))
            pygame.gfxdraw.aacircle(self.screen, int(pos_px[0]), int(pos_px[1]), aura_radius, aura_color)
        
        # Draw player body
        player_rect = pygame.Rect(pos_px[0] - size/2, pos_px[1] - size/2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, 0, 3)
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, player_rect, 2, 3)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p["life"] / 30.0
            size = int(max(1, 5 * life_ratio))
            color = p["color"]
            pygame.draw.rect(self.screen, color, (int(p["pos"][0]), int(p["pos"][1]), size, size))
    
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        # Steps
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))
        # Terraform uses
        terraform_text = self.font_ui.render(f"TERRAFORM: {self.terraform_uses}", True, self.COLOR_UI_TEXT)
        self.screen.blit(terraform_text, (self.SCREEN_WIDTH/2 - terraform_text.get_width()/2, self.SCREEN_HEIGHT - 30))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        win_condition = np.array_equal(self.player_pos, self.data_core_pos)
        text = "CORE REACHED" if win_condition else "DETECTED"
        color = (100, 255, 100) if win_condition else (255, 100, 100)
        
        text_surf = self.font_big.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # We need a screen to display on, separate from the env's internal surface
    pygame.display.init()
    pygame.display.set_caption("Chrome Ghost")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_over_screen = False
    
    # The main loop now uses env.clock for timing
    clock = pygame.time.Clock()
    
    while running:
        movement, space, shift = 0, 0, 0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and game_over_screen:
                game_over_screen = False
                obs, info = env.reset()

        if not game_over_screen:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                game_over_screen = True
                print(f"Game Over! Final Score: {info['score']:.1f}, Steps: {info['steps']}. Press 'R' to restart.")

        # Display the rendered frame from the environment
        frame_to_show = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame_to_show)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()
    pygame.quit()