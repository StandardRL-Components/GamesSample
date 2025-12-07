import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "A puzzle-arcade game where you shoot bubbles to trap enemies, then use the trapped enemies to break bricks."
    user_guide = "Use ← and → arrow keys to aim the launcher. Press space to shoot a bubble."
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # For rendering interpolation, not game logic
        self.SIMULATION_STEPS_PER_TURN = 250  # How long a turn's simulation runs

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_BRICK = [(52, 211, 153), (96, 165, 250), (251, 191, 36)]  # Teal, Blue, Amber
        self.COLOR_ENEMY = (239, 68, 68)
        self.COLOR_BUBBLE = (56, 189, 248)
        self.COLOR_BUBBLE_HIGHLIGHT = (255, 255, 255)
        self.COLOR_TEXT = (229, 231, 235)
        self.COLOR_AIM_LINE = (156, 163, 175)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # Game State Variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.level = None
        self.bricks = None
        self.enemies = None
        self.bubbles = None
        self.particles = None
        self.bubbles_left = None
        self.aim_angle = None
        self.launcher_pos = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1

        self.launcher_pos = (self.WIDTH // 2, self.HEIGHT - 20)
        self.aim_angle = -math.pi / 2  # Pointing straight up

        self.bubbles = []
        self.particles = []

        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        movement = action[0]
        space_pressed = action[1] == 1

        # --- Player Action Phase ---
        aim_speed = 0.08
        if movement == 3:  # Left
            self.aim_angle -= aim_speed
        elif movement == 4:  # Right
            self.aim_angle += aim_speed

        # Clamp angle to prevent aiming downwards
        self.aim_angle = max(-math.pi + 0.1, min(-0.1, self.aim_angle))

        if space_pressed and self.bubbles_left > 0 and not self.bubbles:
            self.bubbles_left -= 1
            bubble_speed = 4.0
            vel = (math.cos(self.aim_angle) * bubble_speed, math.sin(self.aim_angle) * bubble_speed)

            self.bubbles.append({
                "pos": list(self.launcher_pos),
                "vel": list(vel),
                "radius": 10,
                "lifetime": self.SIMULATION_STEPS_PER_TURN,
                "state": "active"  # active, popping
            })

            # --- Simulation Phase ---
            turn_reward = self._run_turn_simulation()
            reward += turn_reward

        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if not any(b for row in self.bricks for b in row if b and b["state"] == "intact"): # Win condition
                reward += 100
            else:  # Lose condition
                reward -= 100

        # Check for level clear and advance
        if not any(b for row in self.bricks for b in row if b and b["state"] == "intact") and not self.game_over:
            self.level += 1
            reward += 50  # Bonus for clearing a level
            self._generate_level()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _run_turn_simulation(self):
        turn_reward = 0
        for _ in range(self.SIMULATION_STEPS_PER_TURN):
            # Update Particles
            for p in self.particles[:]:
                p["pos"][0] += p["vel"][0]
                p["pos"][1] += p["vel"][1]
                p["lifetime"] -= 1
                if p["lifetime"] <= 0:
                    self.particles.remove(p)

            # Update Enemies
            for enemy in self.enemies:
                if not enemy["trapped"]:
                    enemy["pos"][0] += enemy["vel"][0] * enemy["direction"]
                    if not enemy["path_start_x"] <= enemy["pos"][0] <= enemy["path_end_x"]:
                        enemy["direction"] *= -1
                        enemy["pos"][0] = max(enemy["path_start_x"], min(enemy["pos"][0], enemy["path_end_x"]))

            # Update Bubbles
            for bubble in self.bubbles:
                if bubble["state"] == "active":
                    bubble["pos"][0] += bubble["vel"][0]
                    bubble["pos"][1] += bubble["vel"][1]
                    bubble["lifetime"] -= 1

                    # Wall bounce
                    if bubble["pos"][0] <= bubble["radius"] or bubble["pos"][0] >= self.WIDTH - bubble["radius"]:
                        bubble["vel"][0] *= -1
                    if bubble["pos"][1] <= bubble["radius"]:
                        bubble["vel"][1] *= -1

                    if bubble["lifetime"] <= 0 or bubble["pos"][1] >= self.HEIGHT:
                        bubble["state"] = "popping"
                        self._create_particles(bubble["pos"], 10, self.COLOR_BUBBLE)

            # --- Collision Detection ---

            # Bubble-Enemy
            for bubble in self.bubbles:
                if bubble["state"] == "active":
                    for enemy in self.enemies:
                        if not enemy["trapped"]:
                            dist = math.hypot(bubble["pos"][0] - enemy["pos"][0], bubble["pos"][1] - enemy["pos"][1])
                            if dist < bubble["radius"] + enemy["size"] / 2:
                                enemy["trapped"] = True
                                enemy["trapped_pos"] = list(enemy["pos"])
                                bubble["state"] = "popping"
                                turn_reward += 1.0
                                self._create_particles(bubble["pos"], 20, self.COLOR_ENEMY)

            # Trapped Enemy - Brick
            bricks_destroyed_this_frame = []
            for enemy in self.enemies:
                if enemy["trapped"]:
                    enemy_rect = pygame.Rect(enemy["pos"][0] - enemy["size"] / 2, enemy["pos"][1] - enemy["size"] / 2,
                                             enemy["size"], enemy["size"])
                    for r_idx, row in enumerate(self.bricks):
                        for c_idx, brick in enumerate(row):
                            if brick and brick["state"] == "intact":
                                if enemy_rect.colliderect(brick["rect"]):
                                    brick["state"] = "destroyed"
                                    bricks_destroyed_this_frame.append(brick)
                                    turn_reward += 0.1
                                    self._create_particles(brick["rect"].center, 15, brick["color"])

            # Cleanup
            self.bubbles = [b for b in self.bubbles if b["state"] != "popping"]
            self.enemies = [e for e in self.enemies if not e["trapped"]]

            if bricks_destroyed_this_frame and not self.enemies:
                break  # End turn early if last enemy cleared bricks

        self.bubbles.clear()  # Clear any remaining bubbles at end of turn
        return turn_reward

    def _generate_level(self):
        self.bubbles_left = 20

        # Bricks
        self.bricks = []
        brick_w, brick_h = 40, 20
        for r in range(5):
            row = []
            for c in range(14):
                if self.np_random.random() > 0.3:
                    x = c * (brick_w + 5) + 25
                    y = r * (brick_h + 5) + 40
                    color_idx = self.np_random.integers(0, len(self.COLOR_BRICK))
                    color = self.COLOR_BRICK[color_idx]
                    row.append({"rect": pygame.Rect(x, y, brick_w, brick_h), "color": color, "state": "intact"})
                else:
                    row.append(None)  # Empty space
            self.bricks.append(row)

        # Enemies
        self.enemies = []
        num_enemies = 1 + (self.level - 1) // 2
        enemy_speed = 1.0 + 0.05 * ((self.level - 1) // 2)

        for i in range(num_enemies):
            path_y = 200 + i * 30
            self.enemies.append({
                "pos": [self.np_random.integers(100, self.WIDTH - 100, endpoint=True), path_y],
                "vel": [enemy_speed, 0],
                "size": 16,
                "path_start_x": 50,
                "path_end_x": self.WIDTH - 50,
                "direction": self.np_random.choice([-1, 1]),
                "trapped": False,
            })

    def _check_termination(self):
        # Win condition: no bricks left
        if not any(brick["state"] == "intact" for row in self.bricks for brick in row if brick):
            return True
        # Lose condition: out of bubbles and no active bubbles
        if self.bubbles_left <= 0 and not self.bubbles:
            return True
        # Max steps
        if self.steps >= 1000:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]),
                                          int(p["radius"] * (p["lifetime"] / p["max_lifetime"])), p["color"])

        # Render bricks
        for row in self.bricks:
            for brick in row:
                if brick and brick["state"] == "intact":
                    self._draw_rounded_rect(self.screen, brick["rect"], brick["color"], 4)

        # Render enemies
        for enemy in self.enemies:
            if not enemy["trapped"]:
                pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
                size = enemy["size"]
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size // 2, self.COLOR_ENEMY)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size // 2, (255, 150, 150))

        # Render bubbles
        for bubble in self.bubbles:
            pos = (int(bubble["pos"][0]), int(bubble["pos"][1]))
            rad = int(bubble["radius"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], rad, self.COLOR_BUBBLE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], rad, self.COLOR_BUBBLE_HIGHLIGHT)
            # Highlight
            h_pos = (int(pos[0] - rad * 0.4), int(pos[1] - rad * 0.4))
            pygame.gfxdraw.filled_circle(self.screen, h_pos[0], h_pos[1], rad // 4, self.COLOR_BUBBLE_HIGHLIGHT)

    def _render_ui(self):
        # Render score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Render bubbles left
        bubble_icon_rad = 8
        for i in range(self.bubbles_left):
            x = self.WIDTH - 20 - i * (bubble_icon_rad * 2 + 5)
            y = 15
            pygame.gfxdraw.filled_circle(self.screen, x, y, bubble_icon_rad, self.COLOR_BUBBLE)
            pygame.gfxdraw.aacircle(self.screen, x, y, bubble_icon_rad, self.COLOR_BUBBLE_HIGHLIGHT)

        # Render aiming line
        if not self.bubbles:
            length = 60
            end_x = self.launcher_pos[0] + math.cos(self.aim_angle) * length
            end_y = self.launcher_pos[1] + math.sin(self.aim_angle) * length
            self._draw_aa_line_dashed(self.screen, self.launcher_pos, (end_x, end_y), self.COLOR_AIM_LINE, width=2,
                                      dash_length=5)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            lifetime = self.np_random.integers(15, 30, endpoint=True)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "radius": self.np_random.integers(2, 5, endpoint=True),
                "color": color,
                "lifetime": lifetime,
                "max_lifetime": lifetime
            })

    def _draw_rounded_rect(self, surface, rect, color, corner_radius):
        if rect.width < 2 * corner_radius or rect.height < 2 * corner_radius:
            raise ValueError("Rectangle is too small for the corner radius.")
        pygame.draw.rect(surface, color, rect.inflate(-2 * corner_radius, 0))
        pygame.draw.rect(surface, color, rect.inflate(0, -2 * corner_radius))
        pygame.draw.circle(surface, color, (rect.left + corner_radius, rect.top + corner_radius), corner_radius)
        pygame.draw.circle(surface, color, (rect.right - corner_radius - 1, rect.top + corner_radius), corner_radius)
        pygame.draw.circle(surface, color, (rect.left + corner_radius, rect.bottom - corner_radius - 1), corner_radius)
        pygame.draw.circle(surface, color, (rect.right - corner_radius - 1, rect.bottom - corner_radius - 1), corner_radius)

    def _draw_aa_line_dashed(self, surface, p1, p2, color, width=1, dash_length=10):
        x1, y1 = p1
        x2, y2 = p2
        dl = dash_length
        if (x1 == x2 and y1 == y2): return

        line_length = math.hypot(x2 - x1, y2 - y1)
        dashes = int(line_length / dl)

        for i in range(dashes):
            start = i * dl
            end = start + dl / 2

            s_pos = (x1 + (x2 - x1) * start / line_length, y1 + (y2 - y1) * start / line_length)
            e_pos = (x1 + (x2 - x1) * end / line_length, y1 + (y2 - y1) * end / line_length)

            pygame.draw.aaline(surface, color, s_pos, e_pos, width)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bubbles_left": self.bubbles_left,
            "level": self.level
        }

    def close(self):
        pygame.quit()


# Example usage to run and visualize the environment
if __name__ == '__main__':
    # To run with visualization, you may need to comment out the SDL_VIDEODRIVER line at the top
    # and ensure you have a display environment.
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass

    env = GameEnv()
    obs, info = env.reset(seed=42)

    # Use Pygame for human interaction
    pygame.display.set_caption("Bubble Trap Brick Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    running = True
    while running:
        movement_action = 0  # None
        space_action = 0  # Released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4

        if keys[pygame.K_SPACE]:
            space_action = 1

        action = [movement_action, space_action, 0]  # Shift is unused

        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(env.FPS)

    env.close()