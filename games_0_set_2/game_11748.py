import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:28:23.054456
# Source Brief: brief_01748.md
# Brief Index: 1748
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An exploration-puzzle game set in the Forgotten Kingdom of Kush.
    The agent controls a cursor to find artifacts, which are visible under
    different conditions (day/night). Collecting artifacts helps decipher
    prophecies, unlocking new temples and progressing the game.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Cursor Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - actions[1]: Primary Action (Space) (0: released, 1: held) -> Press to collect artifact.
    - actions[2]: Secondary Action (Shift) (0: released, 1: held) -> Press to switch day/night.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Explore the Forgotten Kingdom of Kush, switching between day and night to reveal hidden artifacts. "
        "Collect them all to decipher ancient prophecies and unlock lost temples."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press space to collect a nearby artifact and "
        "shift to switch between day and night."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    UI_HEIGHT = 80
    GAME_HEIGHT = SCREEN_HEIGHT - UI_HEIGHT
    MAX_STEPS = 1000
    CURSOR_SPEED = 15
    ISO_TILE_WIDTH_HALF = 24
    ISO_TILE_HEIGHT_HALF = 12

    # --- COLORS ---
    COLOR_BG_DAY = (135, 206, 235)
    COLOR_BG_NIGHT = (25, 25, 112)
    COLOR_SUN = (255, 223, 0)
    COLOR_MOON = (240, 240, 255)
    COLOR_STAR = (255, 255, 255)
    COLOR_UI_BG = (10, 10, 30, 200)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_UI_TEXT_HIGHLIGHT = (100, 255, 100)
    COLOR_CURSOR = (255, 0, 128)
    COLOR_ARTIFACT = (255, 215, 0)
    COLOR_PROPHECY_COMPLETE = (60, 255, 60)

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
        self.font_ui = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_prophecy = pygame.font.SysFont("georgia", 18, bold=True)
        
        # --- GAME DATA ---
        self._define_game_data()

        # --- STATE VARIABLES ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.is_day = True
        self.current_temple_idx = 0
        self.unlocked_temples = 1
        self.artifact_states = []
        self.prophecy_states = []
        self.prev_action = np.array([0, 0, 0])
        self.particles = []
        self.day_night_transition = 0.0 # 0 = stable, > 0 = transitioning
        self.stars = []

        # --- INITIALIZE AND VALIDATE ---
        # self.reset() # reset is called by the environment runner
        # self.validate_implementation() # Validation is for dev, not production

    def _define_game_data(self):
        self.TEMPLES = [
            {
                "name": "Temple of the Sun", "floor_size": (8, 8),
                "artifacts": [
                    {"id": 0, "name": "Sunstone", "pos": (2, 2, 0), "day": True, "night": False},
                    {"id": 1, "name": "Golden Scarab", "pos": (6, 5, 0), "day": True, "night": True},
                    {"id": 2, "name": "Pharaoh's Scepter", "pos": (1, 6, 0), "day": True, "night": False},
                ]
            },
            {
                "name": "Temple of the Moon", "floor_size": (7, 9),
                "artifacts": [
                    {"id": 3, "name": "Moonstone", "pos": (1, 7, 0), "day": False, "night": True},
                    {"id": 4, "name": "Obsidian Cat", "pos": (5, 2, 0), "day": False, "night": True},
                    {"id": 5, "name": "Silver Ankh", "pos": (3, 4, 0), "day": True, "night": True},
                ]
            },
            {
                "name": "Sanctum of Twilight", "floor_size": (8, 8),
                "artifacts": [
                    {"id": 6, "name": "Duality Mask", "pos": (4, 4, 2), "day": True, "night": False},
                    {"id": 7, "name": "Eclipse Gem", "pos": (4, 4, 2), "day": False, "night": True},
                    {"id": 8, "name": "Eternal Scroll", "pos": (1, 1, 0), "day": True, "night": True},
                ]
            }
        ]
        self.PROPHECIES = [
            {"text": "The Sun's gifts shall reveal the path of Night.", "artifacts_needed": [0, 1, 2], "unlocks_temple": 1},
            {"text": "The Moon's secrets will unlock the final Sanctum.", "artifacts_needed": [3, 4, 5], "unlocks_temple": 2},
            {"text": "When Day and Night unite, the Kingdom is reborn.", "artifacts_needed": [6, 7, 8], "unlocks_temple": -1},
        ]
        self.total_artifacts = sum(len(t['artifacts']) for t in self.TEMPLES)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = [self.SCREEN_WIDTH // 2, self.GAME_HEIGHT // 2]
        self.is_day = True
        self.current_temple_idx = 0
        self.unlocked_temples = 1
        
        self.artifact_states = [False] * self.total_artifacts
        self.prophecy_states = [False] * len(self.PROPHECIES)
        
        self.prev_action = np.array([0, 0, 0])
        self.particles = []
        self.day_night_transition = 0.0
        
        self.stars = [(random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.GAME_HEIGHT)) for _ in range(100)]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.game_over = self.steps >= self.MAX_STEPS
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1

        # --- PROCESS ACTIONS (ON PRESS) ---
        movement, space_action, shift_action = action[0], action[1], action[2]
        space_pressed = space_action == 1 and self.prev_action[1] == 0
        shift_pressed = shift_action == 1 and self.prev_action[2] == 0
        self.prev_action = action

        # 1. Cursor Movement
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED # Up
        if movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED # Down
        if movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED # Left
        if movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GAME_HEIGHT)

        # 2. Teleport / Day-Night Switch (Shift)
        if shift_pressed:
            if self.day_night_transition <= 0: # Prevent spamming
                # // sfx: time_warp
                self.day_night_transition = 1.0
                if self.is_day: # Day -> Night
                    self.is_day = False
                else: # Night -> Day (and teleport)
                    self.is_day = True
                    self.current_temple_idx = (self.current_temple_idx + 1) % self.unlocked_temples

        # 3. Collect Artifact (Space)
        if space_pressed:
            reward += self._attempt_collect_artifact()

        # --- UPDATE GAME STATE ---
        self._update_animations()
        reward += self._check_prophecies()

        # --- CHECK TERMINATION ---
        terminated = self.steps >= self.MAX_STEPS or all(self.prophecy_states)
        truncated = False # Truncation is not used here, only termination.
        if terminated and all(self.prophecy_states):
            reward += 100.0 # Victory bonus
            self.score += 100

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _attempt_collect_artifact(self):
        temple = self.TEMPLES[self.current_temple_idx]
        for artifact in temple["artifacts"]:
            art_id = artifact["id"]
            is_visible = (self.is_day and artifact["day"]) or (not self.is_day and artifact["night"])
            
            if not self.artifact_states[art_id] and is_visible:
                art_screen_pos = self._iso_to_screen(artifact["pos"])
                dist = math.hypot(self.cursor_pos[0] - art_screen_pos[0], self.cursor_pos[1] - art_screen_pos[1])
                
                if dist < 20: # Click radius
                    # // sfx: artifact_collect
                    self.artifact_states[art_id] = True
                    self.score += 1
                    self._spawn_particles(art_screen_pos, self.COLOR_ARTIFACT)
                    return 1.0 # Reward for collecting
        return 0.0

    def _check_prophecies(self):
        reward = 0.0
        for i, prophecy in enumerate(self.PROPHECIES):
            if not self.prophecy_states[i]:
                if all(self.artifact_states[art_id] for art_id in prophecy["artifacts_needed"]):
                    # // sfx: prophecy_reveal
                    self.prophecy_states[i] = True
                    self.score += 10
                    reward += 10.0
                    if prophecy["unlocks_temple"] != -1:
                        self.unlocked_temples = max(self.unlocked_temples, prophecy["unlocks_temple"] + 1)
        return reward

    def _update_animations(self):
        if self.day_night_transition > 0:
            self.day_night_transition -= 0.05
        
        new_particles = []
        for p in self.particles:
            p['life'] -= 1
            if p['life'] > 0:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['radius'] = max(0, p['radius'] * 0.95)
                new_particles.append(p)
        self.particles = new_particles

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "current_temple": self.TEMPLES[self.current_temple_idx]["name"],
            "is_day": self.is_day,
            "unlocked_temples": self.unlocked_temples,
            "prophecies_deciphered": sum(self.prophecy_states)
        }

    # --- RENDERING ---
    def _iso_to_screen(self, pos_3d):
        iso_x, iso_y, iso_z = pos_3d
        screen_x = (self.SCREEN_WIDTH / 2) + (iso_x - iso_y) * self.ISO_TILE_WIDTH_HALF
        screen_y = (self.GAME_HEIGHT / 3) + (iso_x + iso_y) * self.ISO_TILE_HEIGHT_HALF - iso_z * self.ISO_TILE_HEIGHT_HALF * 2
        return int(screen_x), int(screen_y)

    def _lerp_color(self, color1, color2, t):
        t = np.clip(t, 0.0, 1.0)
        return tuple(int(a + (b - a) * t) for a, b in zip(color1, color2))

    def _render_game(self):
        # 1. Background
        transition_t = 0.5 * (1 - math.cos(self.day_night_transition * math.pi)) # Ease in/out
        bg_color = self._lerp_color(self.COLOR_BG_NIGHT, self.COLOR_BG_DAY, transition_t if self.is_day else 1 - transition_t)
        self.screen.fill(bg_color)
        
        # 2. Sky objects (Sun/Moon/Stars)
        sky_obj_alpha = 1 - self.day_night_transition if self.day_night_transition > 0.5 else self.day_night_transition
        if self.is_day:
            pygame.gfxdraw.filled_circle(self.screen, 50, 50, 30, (*self.COLOR_SUN, int(255 * sky_obj_alpha)))
        else:
            pygame.gfxdraw.filled_circle(self.screen, 50, 50, 25, (*self.COLOR_MOON, int(255 * sky_obj_alpha)))
            for x, y in self.stars:
                pygame.gfxdraw.pixel(self.screen, x, y, (*self.COLOR_STAR, int(150 * sky_obj_alpha)))

        # 3. Temple Floor
        temple = self.TEMPLES[self.current_temple_idx]
        w, h = temple["floor_size"]
        for x in range(w):
            for y in range(h):
                p1 = self._iso_to_screen((x, y, 0))
                p2 = self._iso_to_screen((x + 1, y, 0))
                p3 = self._iso_to_screen((x + 1, y + 1, 0))
                p4 = self._iso_to_screen((x, y + 1, 0))
                color = (40, 30, 20) if self.is_day else (20, 15, 10)
                pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3, p4), color)

        # 4. Artifacts
        for artifact in temple["artifacts"]:
            art_id = artifact["id"]
            is_visible = (self.is_day and artifact["day"]) or (not self.is_day and artifact["night"])
            if not self.artifact_states[art_id] and is_visible:
                pos = self._iso_to_screen(artifact["pos"])
                # Glow effect
                glow_radius = int(10 + 5 * math.sin(self.steps * 0.1))
                self._render_glow(pos, glow_radius, (*self.COLOR_ARTIFACT, 50))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_ARTIFACT)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, self.COLOR_ARTIFACT)
        
        # 5. Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), (*p['color'], int(p['life'] * 2.5)))

        # 6. Cursor
        cx, cy = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        radius = int(8 + 2 * math.sin(self.steps * 0.2))
        self._render_glow((cx, cy), radius + 5, (*self.COLOR_CURSOR, 100))
        pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, self.COLOR_CURSOR)
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, 2, self.COLOR_CURSOR)

    def _render_ui(self):
        ui_surface = pygame.Surface((self.SCREEN_WIDTH, self.UI_HEIGHT), pygame.SRCALPHA)
        ui_surface.fill(self.COLOR_UI_BG)
        
        # Left side: Status
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        temple_text = self.font_ui.render(f"LOC: {self.TEMPLES[self.current_temple_idx]['name']}", True, self.COLOR_UI_TEXT)
        ui_surface.blit(score_text, (10, 10))
        ui_surface.blit(steps_text, (10, 30))
        ui_surface.blit(temple_text, (10, 50))

        # Center: Prophecies
        prophecy_y = 10
        for i, prophecy in enumerate(self.PROPHECIES):
            color = self.COLOR_PROPHECY_COMPLETE if self.prophecy_states[i] else self.COLOR_UI_TEXT
            text = f"✓ {prophecy['text']}" if self.prophecy_states[i] else f"□ {prophecy['text']}"
            prophecy_surf = self.font_prophecy.render(text, True, color)
            ui_surface.blit(prophecy_surf, (190, prophecy_y))
            prophecy_y += 22

        # Right side: Controls
        shift_text = self.font_ui.render("[SHIFT] Cycle Time/Temple", True, self.COLOR_UI_TEXT_HIGHLIGHT)
        space_text = self.font_ui.render("[SPACE] Collect Artifact", True, self.COLOR_UI_TEXT_HIGHLIGHT)
        ui_surface.blit(shift_text, (self.SCREEN_WIDTH - shift_text.get_width() - 10, 20))
        ui_surface.blit(space_text, (self.SCREEN_WIDTH - space_text.get_width() - 10, 45))

        self.screen.blit(ui_surface, (0, self.SCREEN_HEIGHT - self.UI_HEIGHT))

    def _render_glow(self, pos, radius, color):
        for i in range(radius, 0, -2):
            alpha = color[3] * (1 - i / radius)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], i, (*color[:3], int(alpha)))

    def _spawn_particles(self, pos, color):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(20, 40),
                'radius': random.uniform(2, 5),
                'color': color
            })
            
    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    # This block will not run in the test environment, but is useful for local testing.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This demonstrates the environment with human input for testing purposes.
    # The agent would use env.step(action) instead.
    
    obs, info = env.reset()
    running = True
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Forgotten Kingdom of Kush")
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # [movement, space, shift]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # --- Map keyboard to action space ---
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: action[0] = 4
        
        # Buttons
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}")

        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']}")
            obs, info = env.reset()

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()