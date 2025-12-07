import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()
pygame.font.init()

from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Stack falling crates using a magnetic hook. "
        "Complete the stack before time runs out, but be careful not to let them collapse or fall off."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to move the hook. "
        "Press space to grab a falling crate and shift to release it."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    TIME_LIMIT_SECONDS = 30

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_PLATFORM = (80, 80, 90)
    COLOR_WALL = (10, 15, 20)
    COLOR_HOOK = (220, 220, 230)
    COLOR_HOOK_GLOW = (220, 220, 230, 50)
    COLOR_CABLE = (150, 150, 160)
    COLOR_TEXT = (230, 230, 230)
    COLOR_SUCCESS = (100, 255, 100)
    COLOR_FAILURE = (255, 100, 100)

    # Physics
    GRAVITY = 0.15
    HOOK_SPEED = 6
    PLATFORM_Y = HEIGHT - 40
    WALL_THICKNESS = 10

    # Crate Properties: (width, height, color)
    CRATE_SPECS = [
        (60, 40, (227, 73, 68)),  # Red
        (80, 30, (79, 153, 224)),  # Blue
        (50, 50, (106, 204, 134)),  # Green
        (70, 35, (247, 211, 88)),  # Yellow
        (65, 45, (172, 111, 219)),  # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 64, bold=True)

        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

        self.hook_x = 0
        self.crates = []
        self.attached_crate_idx = None
        self.particles = []

        self.prev_space_held = False
        self.prev_shift_held = False

        self.max_steps = self.TIME_LIMIT_SECONDS * self.FPS

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

        self.hook_x = self.WIDTH // 2
        self.attached_crate_idx = None
        self.particles = []

        self.prev_space_held = False
        self.prev_shift_held = False

        self._initialize_crates()

        return self._get_observation(), self._get_info()

    def _initialize_crates(self):
        self.crates = []
        for i, (w, h, color) in enumerate(self.CRATE_SPECS):
            self.crates.append({
                "id": i,
                "rect": pygame.Rect(0, 0, w, h),
                "vy": 0.0,
                "color": color,
                "state": "waiting",  # waiting, falling, held, stacked, removed
                "flash_timer": 0,
                "spawn_time": i * 90 + 30  # Staggered spawn
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        self.steps += 1
        reward = 0.0

        self._handle_spawning()
        reward += self._handle_input(movement, space_press, shift_press)
        collision_reward, termination_event = self._update_physics()
        reward += collision_reward

        terminated, terminal_reward, self.win_message = self._check_termination(termination_event)
        reward += terminal_reward
        self.score += reward

        self.game_over = terminated

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        truncated = self.steps >= self.max_steps
        if truncated and not terminated:
            terminated = True # For Gymnasium, truncated implies terminated
            self.win_message = "TIME UP!"
            reward -= 100

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_spawning(self):
        for crate in self.crates:
            if crate["state"] == "waiting" and self.steps >= crate["spawn_time"]:
                crate["state"] = "falling"
                crate["rect"].centerx = self.np_random.integers(
                    self.WALL_THICKNESS + crate["rect"].width // 2,
                    self.WIDTH - self.WALL_THICKNESS - crate["rect"].width // 2
                )
                crate["rect"].y = -crate["rect"].height

    def _handle_input(self, movement, space_press, shift_press):
        reward = 0
        # Horizontal movement
        if movement == 3:  # Left
            self.hook_x -= self.HOOK_SPEED
        elif movement == 4:  # Right
            self.hook_x += self.HOOK_SPEED

        self.hook_x = np.clip(self.hook_x, self.WALL_THICKNESS, self.WIDTH - self.WALL_THICKNESS)

        # Attach crate
        if space_press and self.attached_crate_idx is None:
            # Find highest, non-held, falling crate under the hook
            best_crate = None
            max_y = -1
            for i, crate in enumerate(self.crates):
                if crate["state"] == "falling" and crate["rect"].collidepoint(self.hook_x, crate["rect"].centery):
                    if crate["rect"].y > max_y:
                        max_y = crate["rect"].y
                        best_crate = i

            if best_crate is not None:
                self.attached_crate_idx = best_crate
                self.crates[best_crate]["state"] = "held"
                self.crates[best_crate]["vy"] = 0
                reward += 1

        # Release crate
        elif shift_press and self.attached_crate_idx is not None:
            self.crates[self.attached_crate_idx]["state"] = "falling"
            self.attached_crate_idx = None

        return reward

    def _update_physics(self):
        reward = 0
        termination_event = None

        # Update held crate
        if self.attached_crate_idx is not None:
            crate = self.crates[self.attached_crate_idx]
            crate["rect"].centerx = int(self.hook_x)
            crate["rect"].y = 60  # Fixed height when held

        # Update falling/stacked crates
        for i, crate in enumerate(self.crates):
            if crate["state"] in ["falling", "stacked"]:
                crate["flash_timer"] = max(0, crate["flash_timer"] - 1)

            if crate["state"] == "falling":
                crate["vy"] += self.GRAVITY
                crate["rect"].y += crate["vy"]

                # Check for collision with platform
                if crate["rect"].bottom > self.PLATFORM_Y:
                    crate["rect"].bottom = self.PLATFORM_Y
                    crate["vy"] = 0
                    crate["state"] = "stacked"
                    reward += 5  # Reward for successfully stacking on platform
                    self._create_particles(crate["rect"].midbottom, crate["color"])
                    crate["flash_timer"] = 10

                # Check for collision with other stacked crates
                for j, other_crate in enumerate(self.crates):
                    if i != j and other_crate["state"] == "stacked":
                        if crate["rect"].colliderect(other_crate["rect"]):
                            # 20% collapse chance
                            if self.np_random.random() < 0.20:
                                crate["state"] = "removed"
                                termination_event = "collapse"
                                self._create_particles(crate["rect"].center, self.COLOR_FAILURE, 30)
                                break

                            crate["rect"].bottom = other_crate["rect"].top
                            crate["vy"] = 0
                            crate["state"] = "stacked"
                            reward += 5  # Reward for stacking on another crate
                            self._create_particles(crate["rect"].midbottom, crate["color"])
                            crate["flash_timer"] = 10
                if termination_event:
                    break

        # Check for out of bounds
        for crate in self.crates:
            if crate["state"] == "falling" and (crate["rect"].top > self.HEIGHT or
                                                crate["rect"].left < self.WALL_THICKNESS or
                                                crate["rect"].right > self.WIDTH - self.WALL_THICKNESS):
                if crate["state"] != "removed":  # Only trigger if not already handled by collapse
                    crate["state"] = "removed"
                    termination_event = "fall_off"
                    break

        self._update_particles()
        return reward, termination_event

    def _check_termination(self, event):
        if event == "collapse":
            return True, -50, "CRATE COLLAPSED!"
        if event == "fall_off":
            return True, -50, "CRATE FELL OFF!"

        if self.steps >= self.max_steps:
            return True, -100, "TIME UP!"

        num_stacked = sum(1 for c in self.crates if c["state"] == "stacked")
        if num_stacked == len(self.CRATE_SPECS):
            return True, 100, "SUCCESS!"

        return False, 0, ""

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL,
                         (self.WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.HEIGHT))

        # Platform
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, (0, self.PLATFORM_Y, self.WIDTH, self.HEIGHT - self.PLATFORM_Y))
        pygame.draw.line(self.screen, (200, 200, 210), (0, self.PLATFORM_Y), (self.WIDTH, self.PLATFORM_Y), 2)

        # Particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)

        # Crates
        for crate in self.crates:
            if crate["state"] not in ["waiting", "removed"]:
                self._draw_crate(self.screen, crate)

        # Hook and Cable
        hook_y = 60
        pygame.draw.line(self.screen, self.COLOR_CABLE, (self.hook_x, 0), (self.hook_x, hook_y), 2)

        hook_rect = pygame.Rect(0, 0, 20, 20)
        hook_rect.center = (int(self.hook_x), hook_y)

        # Glow effect
        glow_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_HOOK_GLOW, (20, 20), 20)
        self.screen.blit(glow_surf, (hook_rect.centerx - 20, hook_rect.centery - 20))

        pygame.draw.rect(self.screen, self.COLOR_HOOK, hook_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_BG, hook_rect.inflate(-6, -6), border_radius=3)

    def _draw_crate(self, surface, crate):
        r = crate["rect"]
        color = crate["color"]

        if crate["flash_timer"] > 0:
            flash_amt = crate["flash_timer"] / 10.0
            color = (
                min(255, color[0] + int(150 * flash_amt)),
                min(255, color[1] + int(150 * flash_amt)),
                min(255, color[2] + int(150 * flash_amt)),
            )

        # Main body
        pygame.draw.rect(surface, color, r, border_radius=4)

        # 3D effect
        light_color = (min(255, color[0] + 30), min(255, color[1] + 30), min(255, color[2] + 30))
        dark_color = (max(0, color[0] - 30), max(0, color[1] - 30), max(0, color[2] - 30))

        pygame.draw.line(surface, light_color, r.topleft, r.topright, 3)
        pygame.draw.line(surface, light_color, r.topleft, r.bottomleft, 3)
        pygame.draw.line(surface, dark_color, r.bottomleft, r.bottomright, 3)
        pygame.draw.line(surface, dark_color, r.topright, r.bottomright, 3)

    def _render_ui(self):
        # Timer
        time_left = (self.max_steps - self.steps) / self.FPS
        time_text = self.font_main.render(f"TIME: {max(0, time_left):.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 20, 10))

        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            msg_color = self.COLOR_SUCCESS if "SUCCESS" in self.win_message else self.COLOR_FAILURE
            end_text = self.font_big.render(self.win_message, True, msg_color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, pos, color, count=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.uniform(3, 7),
                'color': color,
                'life': self.np_random.integers(20, 40),
                'max_life': 40
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # particle gravity
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] > 0 and p['radius'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires pygame to be installed and will open a window.
    # The environment itself is headless and does not require a display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit() # a bit of a hack to re-init with video
    pygame.init()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Crate Stacker")
    clock = pygame.time.Clock()

    terminated = False
    truncated = False

    while not terminated and not truncated:
        movement, space_held, shift_held = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space_held = 1

        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Reason: {env.win_message}")
            pygame.time.wait(3000)  # Pause for 3 seconds before closing

        clock.tick(GameEnv.FPS)

    env.close()