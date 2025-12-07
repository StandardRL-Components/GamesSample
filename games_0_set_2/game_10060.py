import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:44:05.094320
# Source Brief: brief_00060.md
# Brief Index: 60
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Manage a swarm of drones, collecting fuel canisters to keep them all powered and survive for the duration."
    user_guide = "Use arrow keys to move the selected drone. Press space to cycle to the next drone."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 60

    # Game parameters
    NUM_DRONES = 8
    NUM_CANISTERS = 10
    MAX_STEPS = 60 * FPS  # 60 seconds

    DRONE_SIZE = 12
    DRONE_SPEED = 3.5
    SELECTED_OUTLINE_WIDTH = 2

    CANISTER_SIZE = 8
    CANISTER_RESPAWN_DELAY = 0.5 * FPS  # 30 frames

    MAX_FUEL = 15.0 * FPS  # 15 seconds of fuel
    INITIAL_FUEL = 5.0 * FPS  # 5 seconds
    FUEL_REFILL_AMOUNT = 7.0 * FPS  # 7 seconds
    FUEL_CONSUMPTION_RATE = 1.0
    LOW_FUEL_THRESHOLD = 2.0 * FPS  # 2 seconds

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_CANISTER = (255, 220, 0)  # Bright Yellow
    COLOR_TEXT = (240, 240, 240)
    COLOR_UI_BG = (40, 50, 70, 180)

    DRONE_COLORS = [
        (52, 152, 219), (231, 76, 60), (46, 204, 113), (155, 89, 182),
        (241, 196, 15), (26, 188, 156), (230, 126, 34), (52, 73, 94)
    ]

    # Rewards
    REWARD_COLLECT = 0.1
    REWARD_SURVIVE_TICK = 1.0 / FPS
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 14)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.drones = []
        self.canisters = []
        self.canister_respawn_queue = []
        self.particles = []
        self.selected_drone_idx = 0
        self.last_space_held = False

        # self.reset() is called by the gym wrapper
        
        # Critical self-check - not part of standard API, but useful for dev
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.selected_drone_idx = 0
        self.last_space_held = False

        # Initialize Drones
        self.drones = []
        for i in range(self.NUM_DRONES):
            self.drones.append({
                'pos': pygame.Vector2(
                    self.np_random.uniform(self.DRONE_SIZE, self.SCREEN_WIDTH - self.DRONE_SIZE),
                    self.np_random.uniform(self.DRONE_SIZE, self.SCREEN_HEIGHT - self.DRONE_SIZE)
                ),
                'fuel': self.INITIAL_FUEL,
                'color': self.DRONE_COLORS[i % len(self.DRONE_COLORS)]
            })

        # Initialize Fuel Canisters
        self.canisters = []
        for _ in range(self.NUM_CANISTERS):
            self._spawn_canister()

        self.canister_respawn_queue = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- 1. Unpack and handle actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Handle drone selection (on space press, not hold)
        if space_held and not self.last_space_held:
            self.selected_drone_idx = (self.selected_drone_idx + 1) % self.NUM_DRONES
            # sfx: UI_Switch.wav
        self.last_space_held = space_held

        # --- 2. Update game logic ---
        # Move selected drone
        selected_drone = self.drones[self.selected_drone_idx]
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1  # Up
        elif movement == 2: move_vec.y = 1   # Down
        elif movement == 3: move_vec.x = -1  # Left
        elif movement == 4: move_vec.x = 1   # Right

        if move_vec.length() > 0:
            selected_drone['pos'] += move_vec.normalize() * self.DRONE_SPEED

        # Clamp drone position to screen bounds
        selected_drone['pos'].x = np.clip(selected_drone['pos'].x, self.DRONE_SIZE / 2, self.SCREEN_WIDTH - self.DRONE_SIZE / 2)
        selected_drone['pos'].y = np.clip(selected_drone['pos'].y, self.DRONE_SIZE / 2, self.SCREEN_HEIGHT - self.DRONE_SIZE / 2)

        # Update all drones' fuel
        for drone in self.drones:
            drone['fuel'] = max(0, drone['fuel'] - self.FUEL_CONSUMPTION_RATE)

        # Check for drone-canister collision
        collected_canisters = []
        for i, canister in enumerate(self.canisters):
            dist = selected_drone['pos'].distance_to(canister['pos'])
            if dist < (self.DRONE_SIZE / 2 + self.CANISTER_SIZE / 2):
                selected_drone['fuel'] = min(self.MAX_FUEL, selected_drone['fuel'] + self.FUEL_REFILL_AMOUNT)
                reward += self.REWARD_COLLECT
                collected_canisters.append(i)
                self.canister_respawn_queue.append(self.steps + self.CANISTER_RESPAWN_DELAY)
                self._create_particles(canister['pos'], self.COLOR_CANISTER)
                # sfx: Collect_Fuel.wav
                break  # Only collect one per step

        # Remove collected canisters
        for i in sorted(collected_canisters, reverse=True):
            del self.canisters[i]

        # Respawn canisters
        if self.canister_respawn_queue and self.steps >= self.canister_respawn_queue[0]:
            self.canister_respawn_queue.pop(0)
            self._spawn_canister()

        # Update particles
        self._update_particles()

        # --- 3. Calculate rewards & check termination ---
        terminated = False

        # Survival reward
        reward += self.REWARD_SURVIVE_TICK

        # Check for loss condition (any drone runs out of fuel)
        for drone in self.drones:
            if drone['fuel'] <= 0:
                reward += self.REWARD_LOSE
                terminated = True
                self.game_over = True
                # sfx: Game_Over_Lose.wav
                break

        # Check for win condition (time limit reached)
        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            reward += self.REWARD_WIN
            terminated = True
            self.game_over = True
            # sfx: Game_Over_Win.wav

        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_canister(self):
        # Ensure new canister doesn't spawn on top of an existing one
        while True:
            new_pos = pygame.Vector2(
                self.np_random.uniform(self.CANISTER_SIZE, self.SCREEN_WIDTH - self.CANISTER_SIZE),
                self.np_random.uniform(self.CANISTER_SIZE, self.SCREEN_HEIGHT - self.CANISTER_SIZE)
            )
            is_valid = True
            for canister in self.canisters:
                if new_pos.distance_to(canister['pos']) < self.CANISTER_SIZE * 3:
                    is_valid = False
                    break
            if is_valid:
                self.canisters.append({'pos': new_pos})
                break

    def _get_observation(self):
        # --- Render all game elements ---
        self.screen.fill(self.COLOR_BG)
        self._render_particles()
        self._render_canisters()
        self._render_drones()
        self._render_ui()

        # Convert to numpy array (required format)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_drones(self):
        for i, drone in enumerate(self.drones):
            pos = drone['pos']
            color = drone['color']

            # Low fuel blink effect
            if drone['fuel'] < self.LOW_FUEL_THRESHOLD and (self.steps // 6) % 2 == 0:
                color = (255, 0, 0)

            drone_rect = pygame.Rect(
                int(pos.x - self.DRONE_SIZE / 2), int(pos.y - self.DRONE_SIZE / 2),
                self.DRONE_SIZE, self.DRONE_SIZE
            )

            if i == self.selected_drone_idx:
                # Pulsating glow for selected drone
                glow_size = self.DRONE_SIZE * 1.5 + 4 * math.sin(self.steps * 0.15)
                glow_alpha = 100 + 40 * math.sin(self.steps * 0.15)
                glow_color = (*color, glow_alpha)

                # Draw the glow using gfxdraw for anti-aliasing
                glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, glow_color, (glow_size, glow_size), glow_size)
                self.screen.blit(glow_surf, (int(pos.x - glow_size), int(pos.y - glow_size)), special_flags=pygame.BLEND_RGBA_ADD)

                # Draw drone and outline
                pygame.draw.rect(self.screen, color, drone_rect, border_radius=2)
                pygame.draw.rect(self.screen, (255, 255, 255), drone_rect, self.SELECTED_OUTLINE_WIDTH, border_radius=2)
            else:
                pygame.draw.rect(self.screen, color, drone_rect, border_radius=2)

            # Draw fuel bar above drone
            fuel_pct = drone['fuel'] / self.MAX_FUEL
            bar_width = self.DRONE_SIZE * 2
            bar_height = 4
            bar_x = int(pos.x - bar_width / 2)
            bar_y = int(pos.y - self.DRONE_SIZE)

            fuel_color = (46, 204, 113)  # Green
            if fuel_pct < 0.5: fuel_color = (241, 196, 15)  # Yellow
            if fuel_pct < 0.2: fuel_color = (231, 76, 60)  # Red

            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), border_radius=1)
            pygame.draw.rect(self.screen, fuel_color, (bar_x, bar_y, int(bar_width * fuel_pct), bar_height), border_radius=1)

    def _render_canisters(self):
        for canister in self.canisters:
            pos = (int(canister['pos'].x), int(canister['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, *pos, self.CANISTER_SIZE, self.COLOR_CANISTER)
            pygame.gfxdraw.aacircle(self.screen, *pos, self.CANISTER_SIZE, self.COLOR_CANISTER)

    def _render_ui(self):
        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps) / self.FPS
        timer_text = f"TIME: {int(time_left // 60):02}:{int(time_left % 60):02}"
        timer_surf = self.font_main.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (10, 10))

        # Score
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_surf, score_rect)

        # Game Over Message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))

            win_condition = self.steps >= self.MAX_STEPS
            msg = "MISSION SUCCESS" if win_condition else "DRONE LOST"
            color = (100, 255, 100) if win_condition else (255, 100, 100)

            msg_surf = self.font_main.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(msg_surf, msg_rect)

            final_score_text = f"Final Score: {self.score:.1f}"
            final_score_surf = self.font_small.render(final_score_text, True, self.COLOR_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(final_score_surf, final_score_rect)

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': velocity,
                'lifetime': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95  # Damping
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifetime'] / 30.0))))
            color_with_alpha = (*p['color'], alpha)

            # Using a surface for alpha blending
            particle_surf = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color_with_alpha, (p['size'], p['size']), p['size'])
            self.screen.blit(particle_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "selected_drone": self.selected_drone_idx,
            "drone_fuels": [d['fuel'] for d in self.drones],
        }

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

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # For this to work, you must comment out the `os.environ` line at the top
    # and have pygame installed.
    # os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.

    env = GameEnv()
    obs, info = env.reset()

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Drone Swarm Manager")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement = 0  # No-op
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

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            pygame.time.wait(2000)  # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    env.close()