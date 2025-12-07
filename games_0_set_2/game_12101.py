import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:05:35.705390
# Source Brief: brief_02101.md
# Brief Index: 2101
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls magnetic fish to defend a
    river from encroaching pollution. The core mechanic involves positioning fish
    and triggering their magnetic pulses to create cleansing chain reactions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control magnetic fish to defend a river from encroaching pollution. "
        "Trigger magnetic pulses to create cleansing chain reactions and save the ecosystem."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selected fish. Press space to trigger a magnetic pulse. "
        "Press shift to switch between different fish."
    )
    auto_advance = True

    # --- Class-level state for persistent progression ---
    # This persists across resets but not program executions.
    UNLOCKED_FISH_SPECIES = [
        {"color": (255, 80, 80), "pulse_range": 80, "pulse_strength": 1.0, "cooldown": 60, "name": "Red Fugu"},
    ]
    ALL_FISH_SPECIES = [
        {"color": (255, 80, 80), "pulse_range": 80, "pulse_strength": 1.0, "cooldown": 60, "name": "Red Fugu"},
        {"color": (80, 255, 80), "pulse_range": 100, "pulse_strength": 1.2, "cooldown": 75, "name": "Green Guppy"},
        {"color": (80, 80, 255), "pulse_range": 60, "pulse_strength": 1.5, "cooldown": 50, "name": "Blue Betta"},
    ]
    SUCCESSFUL_CLEARS = 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000

        # --- Colors ---
        self.COLOR_BG_DARK = (10, 20, 40)
        self.COLOR_BG_LIGHT = (40, 80, 150)
        self.COLOR_POLLUTION = (80, 40, 90)
        self.COLOR_SOURCE = (180, 220, 255)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_UI_SHADOW = (20, 20, 20)
        self.COLOR_SELECTED_GLOW = (255, 255, 150)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.last_shift_state = 0
        self.selected_fish_index = 0
        
        self.fish = []
        self.pollution_blobs = []
        self.pulses = []
        self.particles = []

        self.pollution_spawn_timer = 0
        self.initial_pollution_blobs = 15
        self.pollution_spawn_rate = 90 # Lower is faster
        self.total_initial_pollution = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Core State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.last_shift_state = 0
        self.selected_fish_index = 0

        # --- Reset Game Entities ---
        self.fish.clear()
        self.pollution_blobs.clear()
        self.pulses.clear()
        self.particles.clear()
        
        # --- Initialize Fish ---
        for i, species in enumerate(self.UNLOCKED_FISH_SPECIES):
            self.fish.append({
                "pos": pygame.math.Vector2(self.WIDTH / 2 + (i - (len(self.UNLOCKED_FISH_SPECIES)-1)/2) * 80, self.HEIGHT / 2),
                "vel": pygame.math.Vector2(0, 0),
                "species": species,
                "cooldown_timer": 0,
                "pulse_triggered_by_chain": False
            })

        # --- Initialize Pollution ---
        self.total_initial_pollution = self.initial_pollution_blobs
        for _ in range(self.initial_pollution_blobs):
            self.pollution_blobs.append({
                "pos": pygame.math.Vector2(random.uniform(20, self.WIDTH - 20), random.uniform(self.HEIGHT * 0.6, self.HEIGHT - 20)),
                "radius": random.uniform(8, 15),
                "speed": random.uniform(0.3, 0.7)
            })

        # --- Reset Timers & Progression ---
        self.pollution_spawn_timer = 0
        self.pollution_spawn_rate = 90
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_this_step = 0 # Small penalty for existing

        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_fish()
        self._update_pollution()
        self._update_pulses()
        self._update_particles()
        self._update_progression()

        # --- Check Termination Conditions ---
        terminated = self._check_termination()

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Fish Selection (Shift) ---
        if shift_held and not self.last_shift_state and len(self.fish) > 1:
            self.selected_fish_index = (self.selected_fish_index + 1) % len(self.fish)
            # SFX: UI_SWITCH
        self.last_shift_state = shift_held

        # --- Fish Movement ---
        if self.fish:
            selected_fish = self.fish[self.selected_fish_index]
            speed = 4
            if movement == 1: selected_fish["vel"].y = -speed # Up
            elif movement == 2: selected_fish["vel"].y = speed # Down
            elif movement == 3: selected_fish["vel"].x = -speed # Left
            elif movement == 4: selected_fish["vel"].x = speed # Right
            else: selected_fish["vel"] = pygame.math.Vector2(0, 0)

        # --- Magnetic Pulse (Space) ---
        if space_held and self.fish:
            self._trigger_pulse(self.selected_fish_index)

    def _trigger_pulse(self, fish_index, is_chain_reaction=False):
        fish = self.fish[fish_index]
        if fish["cooldown_timer"] == 0:
            fish["cooldown_timer"] = fish["species"]["cooldown"]
            self.pulses.append({
                "pos": fish["pos"].copy(),
                "radius": 0,
                "max_radius": fish["species"]["pulse_range"],
                "strength": fish["species"]["pulse_strength"],
                "color": fish["species"]["color"],
                "hit_blobs": set(),
                "hit_fish": {fish_index}
            })
            if is_chain_reaction:
                # SFX: CHAIN_PULSE
                pass
            else:
                # SFX: PLAYER_PULSE
                pass
            
    def _update_fish(self):
        for fish in self.fish:
            # Update position with velocity
            fish["pos"] += fish["vel"]
            # Clamp position to screen bounds
            fish["pos"].x = max(15, min(self.WIDTH - 15, fish["pos"].x))
            fish["pos"].y = max(15, min(self.HEIGHT - 15, fish["pos"].y))
            # Update cooldown
            if fish["cooldown_timer"] > 0:
                fish["cooldown_timer"] -= 1

    def _update_pollution(self):
        for blob in self.pollution_blobs[:]:
            blob["pos"].y -= blob["speed"]
            if blob["pos"].y < 10: # Reached the source
                self.pollution_blobs.remove(blob)
                self.reward_this_step -= 50 # Large penalty
                self.game_over = True
                # SFX: GAME_OVER_POLLUTION
        
        # Spawn new pollution
        self.pollution_spawn_timer += 1
        if self.pollution_spawn_timer > self.pollution_spawn_rate:
            self.pollution_spawn_timer = 0
            self.pollution_blobs.append({
                "pos": pygame.math.Vector2(random.uniform(20, self.WIDTH - 20), self.HEIGHT + 20),
                "radius": random.uniform(8, 15),
                "speed": random.uniform(0.3, 0.7)
            })

    def _update_pulses(self):
        for pulse in self.pulses[:]:
            # Expand pulse
            pulse["radius"] += 3
            if pulse["radius"] > pulse["max_radius"]:
                self.pulses.remove(pulse)
                continue

            # Check collision with pollution
            for blob in self.pollution_blobs[:]:
                blob_id = id(blob)
                if blob_id in pulse["hit_blobs"]: continue
                
                dist = pulse["pos"].distance_to(blob["pos"])
                if dist < pulse["radius"] + blob["radius"]:
                    # Repel or destroy blob
                    repel_strength = (1 - (dist / (pulse["radius"] + blob["radius"]))) * pulse["strength"] * 5
                    if repel_strength > 1.5: # Destroy
                        self.pollution_blobs.remove(blob)
                        self.reward_this_step += 1.0 # Reward for destroying
                        self._create_particles(blob["pos"], (200, 255, 255), 20)
                        # SFX: POLLUTION_DESTROY
                    else: # Repel
                        direction = (blob["pos"] - pulse["pos"]).normalize()
                        blob["pos"] += direction * repel_strength
                        self.reward_this_step += 0.1 # Reward for repelling
                    pulse["hit_blobs"].add(blob_id)

            # Check collision with other fish for chain reaction
            for i, fish in enumerate(self.fish):
                if i in pulse["hit_fish"]: continue
                dist = pulse["pos"].distance_to(fish["pos"])
                if dist < pulse["radius"]:
                    self._trigger_pulse(i, is_chain_reaction=True)
                    pulse["hit_fish"].add(i)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.math.Vector2(random.uniform(-2, 2), random.uniform(-2, 2)),
                "radius": random.uniform(1, 3),
                "lifespan": random.randint(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _update_progression(self):
        # Increase pollution spawn rate over time
        if self.steps > 0 and self.steps % 300 == 0:
            self.pollution_spawn_rate = max(20, self.pollution_spawn_rate * 0.95)

    def _check_termination(self):
        if self.game_over:
            return True
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True

        if not self.pollution_blobs:
            self.game_over = True
            self.reward_this_step += 50 # Big reward for winning
            # SFX: LEVEL_CLEAR
            GameEnv.SUCCESSFUL_CLEARS += 1
            if GameEnv.SUCCESSFUL_CLEARS >= 2 and len(self.UNLOCKED_FISH_SPECIES) < len(self.ALL_FISH_SPECIES):
                self.UNLOCKED_FISH_SPECIES.append(self.ALL_FISH_SPECIES[len(self.UNLOCKED_FISH_SPECIES)])
                GameEnv.SUCCESSFUL_CLEARS = 0 # Reset for next unlock
            return True

        return False

    def _get_observation(self):
        # --- Render everything ---
        self._render_background()
        self._render_pollution()
        self._render_pulses()
        self._render_particles()
        self._render_fish()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Gradient background
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_DARK[0] * (1 - ratio) + self.COLOR_BG_LIGHT[0] * ratio),
                int(self.COLOR_BG_DARK[1] * (1 - ratio) + self.COLOR_BG_LIGHT[1] * ratio),
                int(self.COLOR_BG_DARK[2] * (1 - ratio) + self.COLOR_BG_LIGHT[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        # River Source (Top)
        for i in range(20):
            alpha = 100 - i * 5
            radius = 60 + i * 4
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH // 2, 0, radius, (*self.COLOR_SOURCE, alpha))

        # Pollution Source (Bottom)
        for i in range(15):
            alpha = 50 - i * 3
            radius = 80 + i * 5 + int(math.sin(self.steps / 10 + i) * 10)
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH // 2, self.HEIGHT, radius, (*self.COLOR_POLLUTION, alpha))

    def _render_pollution(self):
        for blob in self.pollution_blobs:
            wobble = math.sin(self.steps / 5 + blob["pos"].x) * 2
            pygame.gfxdraw.filled_circle(self.screen, int(blob["pos"].x), int(blob["pos"].y), int(blob["radius"] + wobble), self.COLOR_POLLUTION)
            pygame.gfxdraw.aacircle(self.screen, int(blob["pos"].x), int(blob["pos"].y), int(blob["radius"] + wobble), self.COLOR_POLLUTION)

    def _render_fish(self):
        for i, fish in enumerate(self.fish):
            pos = (int(fish["pos"].x), int(fish["pos"].y))
            color = fish["species"]["color"]
            
            # Draw glow for selected fish
            if i == self.selected_fish_index:
                glow_radius = 20 + int(math.sin(self.steps / 6) * 4)
                for j in range(glow_radius, 0, -2):
                    alpha = 60 * (1 - j / glow_radius)
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], j, (*self.COLOR_SELECTED_GLOW, alpha))
            
            # Draw fish body (triangle)
            p1 = (pos[0], pos[1] - 10)
            p2 = (pos[0] - 7, pos[1] + 7)
            p3 = (pos[0] + 7, pos[1] + 7)
            pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], color)
            pygame.gfxdraw.aatrigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], color)

            # Draw cooldown indicator
            if fish["cooldown_timer"] > 0:
                ratio = fish["cooldown_timer"] / fish["species"]["cooldown"]
                pygame.draw.arc(self.screen, (255,255,255), (pos[0]-12, pos[1]-12, 24, 24), -math.pi/2, -math.pi/2 + (2 * math.pi * ratio), 2)

    def _render_pulses(self):
        for pulse in self.pulses:
            ratio = pulse["radius"] / pulse["max_radius"]
            current_radius = int(pulse["radius"])
            alpha = int(150 * (1 - ratio**2))
            if alpha > 0 and current_radius > 0:
                pygame.gfxdraw.aacircle(self.screen, int(pulse["pos"].x), int(pulse["pos"].y), current_radius, (*pulse["color"], alpha))
                pygame.gfxdraw.aacircle(self.screen, int(pulse["pos"].x), int(pulse["pos"].y), current_radius-1, (*pulse["color"], alpha))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, pos, color, shadow_color):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surf, pos)

        # Pollution %
        pollution_remaining = len(self.pollution_blobs)
        pollution_percent = (pollution_remaining / self.total_initial_pollution) * 100 if self.total_initial_pollution > 0 else 0
        draw_text(f"Pollution: {pollution_percent:.1f}%", self.font_main, (10, 10), self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW)
        
        # Score
        draw_text(f"Score: {int(self.score)}", self.font_main, (10, 40), self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW)

        # Timer bar
        time_ratio = self.steps / self.MAX_STEPS
        bar_width = 200
        bar_height = 20
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 10
        pygame.draw.rect(self.screen, self.COLOR_UI_SHADOW, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_SOURCE, (bar_x, bar_y, bar_width * (1 - time_ratio), bar_height))
        
        # Selected Fish Info
        if self.fish:
            fish = self.fish[self.selected_fish_index]
            draw_text(f"Selected: {fish['species']['name']}", self.font_small, (bar_x, bar_y + 25), fish['species']['color'], self.COLOR_UI_SHADOW)


    def _get_info(self):
        self.score += self.reward_this_step
        return {
            "score": self.score,
            "steps": self.steps,
            "pollution_cleared": 1.0 - (len(self.pollution_blobs) / self.total_initial_pollution if self.total_initial_pollution > 0 else 0),
            "unlocked_fish": len(self.UNLOCKED_FISH_SPECIES)
        }

    def close(self):
        pygame.quit()


# --- Example Usage ---
if __name__ == '__main__':
    # The validation code was removed from the original snippet as it's not needed for the final environment.
    # It's good practice for development but not for the final deliverable.
    # If you want to run the validation, you can uncomment the following lines.
    # try:
    #     env_to_validate = GameEnv()
    #     # This is a simplified validation. A more robust one would be in a separate test file.
    #     assert env_to_validate.action_space.shape == (3,)
    #     assert env_to_validate.action_space.nvec.tolist() == [5, 2, 2]
    #     test_obs = env_to_validate._get_observation()
    #     assert test_obs.shape == (env_to_validate.HEIGHT, env_to_validate.WIDTH, 3)
    #     assert test_obs.dtype == np.uint8
    #     obs, info = env_to_validate.reset()
    #     assert obs.shape == (env_to_validate.HEIGHT, env_to_validate.WIDTH, 3)
    #     test_action = env_to_validate.action_space.sample()
    #     obs, reward, term, trunc, info = env_to_validate.step(test_action)
    #     assert obs.shape == (env_to_validate.HEIGHT, env_to_validate.WIDTH, 3)
    #     env_to_validate.close()
    #     print("✓ Implementation appears valid.")
    # except Exception as e:
    #     print(f"✗ Implementation validation failed: {e}")


    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    # This part requires a display. If you run this in a headless environment,
    # it will fail unless you also remove this visualization part.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    pygame.display.init()
    pygame.display.set_caption("Magnetic Fish Defender")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # --- Action Mapping for Human Player ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering for Human ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # Wait a bit before reset
            pygame.time.wait(2000)
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        clock.tick(env.FPS)
        
    env.close()