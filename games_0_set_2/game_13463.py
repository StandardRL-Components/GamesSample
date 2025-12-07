import gymnasium as gym
import os
import pygame
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:51:08.286069
# Source Brief: brief_03463.md
# Brief Index: 3463
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random


class GameEnv(gym.Env):
    """
    A fast-paced, momentum-based puzzle game where the player flicks colored orbs
    into matching slots while contending with shifting gravity and unpredictable obstacles.
    """

    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Flick colored orbs into matching slots while contending with shifting gravity and "
        "unpredictable obstacles."
    )
    user_guide = "Controls: Use arrow keys (↑↓←→) to aim the orb. Press space to launch."
    auto_advance = True

    # --- Static Class Attributes for Persistent State ---
    UNLOCKED_COLORS = {"red", "green", "blue", "yellow"}
    TOTAL_SLOTS_FILLED_EVER = 0

    # --- Game Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (15, 18, 28)
    COLOR_GRID = (30, 35, 50)
    COLOR_OBSTACLE = (80, 80, 90)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_MOMENTUM_BAR = (60, 200, 255)
    COLOR_MOMENTUM_BAR_BG = (40, 40, 60)
    COLOR_GRAVITY = (255, 255, 255, 100)

    COLOR_MAP = {
        "red": (255, 80, 80),
        "green": (80, 255, 80),
        "blue": (80, 120, 255),
        "yellow": (255, 255, 80),
        "purple": (180, 80, 255),
        "orange": (255, 160, 80),
    }

    # Physics & Gameplay
    GRAVITY_STRENGTH = 0.15
    AIM_ADJUST_SPEED = 3.0
    MAX_AIM_POWER = 150
    FLICK_POWER_MULTIPLIER = 0.1
    BOUNCE_DAMPING = 0.85
    FRICTION = 0.995
    MIN_VELOCITY_FOR_MOVING = 0.1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- State Variables Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.momentum = 0.0
        self.orb = None
        self.slots = []
        self.obstacles = []
        self.particles = []
        self.orb_state = "aiming"  # 'aiming' or 'moving'
        self.aim_vector = pygame.Vector2(0, 0)
        self.gravity = pygame.Vector2(0, self.GRAVITY_STRENGTH)
        self.gravity_change_timer = 0
        self.obstacle_spawn_rate = 0.0
        self.obstacle_timer = 0
        self.prev_space_held = False
        self.prev_orb_dist_to_target = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.momentum = 100.0
        self.orb_state = "aiming"
        self.aim_vector = pygame.Vector2(0, 0)
        self.gravity = pygame.Vector2(0, self.GRAVITY_STRENGTH)
        self.gravity_change_timer = 20 * self.FPS  # Change every 20 seconds
        self.obstacle_spawn_rate = 0.1
        self.obstacle_timer = 50 * self.FPS  # Increase rate every 50 seconds
        self.prev_space_held = False
        self.particles.clear()

        self._generate_level()
        self._spawn_orb()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # We need to track which slots are filled in this step for rewards
        for slot in self.slots:
            slot["filled_this_step"] = False

        self._handle_input(action)
        self._update_game_logic()

        reward = self._calculate_reward()
        self.score += reward

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if truncated and not terminated:
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if self.orb_state == "aiming":
            if movement == 1:
                self.aim_vector.y -= self.AIM_ADJUST_SPEED
            elif movement == 2:
                self.aim_vector.y += self.AIM_ADJUST_SPEED
            elif movement == 3:
                self.aim_vector.x -= self.AIM_ADJUST_SPEED
            elif movement == 4:
                self.aim_vector.x += self.AIM_ADJUST_SPEED

            # Clamp aim vector
            if self.aim_vector.length() > self.MAX_AIM_POWER:
                self.aim_vector.scale_to_length(self.MAX_AIM_POWER)

            # Launch on space PRESS
            if space_held and not self.prev_space_held:
                flick_power = self.aim_vector.length()
                if flick_power > 1:
                    # sfx: launch_orb.wav
                    self.orb["vel"] = self.aim_vector * self.FLICK_POWER_MULTIPLIER
                    self.momentum -= 5 + (flick_power / self.MAX_AIM_POWER) * 10
                    self.orb_state = "moving"
                    self.aim_vector.update(0, 0)

        self.prev_space_held = space_held

    def _update_game_logic(self):
        # --- Update Orb Physics ---
        if self.orb and self.orb_state == "moving":
            self.orb["vel"] += self.gravity
            self.orb["vel"] *= self.FRICTION
            self.orb["pos"] += self.orb["vel"]
            self._handle_collisions()

            if self.orb and self.orb["vel"].length() < self.MIN_VELOCITY_FOR_MOVING:
                self.orb["vel"].update(0, 0)
                self.orb_state = "aiming"

        # --- Update Timers & World State ---
        self.gravity_change_timer -= 1
        if self.gravity_change_timer <= 0:
            self.gravity_change_timer = 20 * self.FPS
            directions = [
                pygame.Vector2(0, 1),
                pygame.Vector2(0, -1),
                pygame.Vector2(1, 0),
                pygame.Vector2(-1, 0),
            ]
            self.gravity = random.choice(directions) * self.GRAVITY_STRENGTH
            # sfx: gravity_shift.wav

        self.obstacle_timer -= 1
        if self.obstacle_timer <= 0:
            self.obstacle_timer = 50 * self.FPS
            self.obstacle_spawn_rate = min(0.5, self.obstacle_spawn_rate + 0.01)

        if self.np_random.random() < self.obstacle_spawn_rate / self.FPS:
            self._toggle_obstacle()

        # --- Update Particles ---
        self._update_particles()

    def _handle_collisions(self):
        if not self.orb:
            return
            
        orb_pos, orb_vel, orb_rad = self.orb["pos"], self.orb["vel"], self.orb["radius"]

        # Wall collisions
        if orb_pos.x < orb_rad:
            orb_pos.x = orb_rad
            orb_vel.x *= -self.BOUNCE_DAMPING
        elif orb_pos.x > self.WIDTH - orb_rad:
            orb_pos.x = self.WIDTH - orb_rad
            orb_vel.x *= -self.BOUNCE_DAMPING

        if orb_pos.y < orb_rad:
            orb_pos.y = orb_rad
            orb_vel.y *= -self.BOUNCE_DAMPING
        elif orb_pos.y > self.HEIGHT - orb_rad:
            orb_pos.y = self.HEIGHT - orb_rad
            orb_vel.y *= -self.BOUNCE_DAMPING

        # Obstacle collisions
        for obs_rect in self.obstacles:
            if obs_rect.collidepoint(orb_pos.x, orb_pos.y):
                # Simple push-out and bounce logic
                dx = orb_pos.x - obs_rect.centerx
                dy = orb_pos.y - obs_rect.centery
                if abs(dx) > abs(dy):  # Horizontal collision
                    orb_vel.x *= -self.BOUNCE_DAMPING
                    orb_pos.x = (
                        obs_rect.right + orb_rad if dx > 0 else obs_rect.left - orb_rad
                    )
                else:  # Vertical collision
                    orb_vel.y *= -self.BOUNCE_DAMPING
                    orb_pos.y = (
                        obs_rect.bottom + orb_rad if dy > 0 else obs_rect.top - orb_rad
                    )
                # sfx: bounce.wav

        # Slot collisions
        for slot in self.slots:
            if not slot["filled"] and slot["color_name"] == self.orb["color_name"]:
                dist = orb_pos.distance_to(slot["pos"])
                if dist < slot["radius"] + orb_rad:
                    # sfx: slot_fill.wav
                    slot["filled"] = True
                    slot["filled_this_step"] = True # For reward calculation
                    GameEnv.TOTAL_SLOTS_FILLED_EVER += 1
                    self._create_particles(slot["pos"], slot["color"])
                    self._check_unlocks()
                    self._spawn_orb()
                    return  # Exit to avoid issues with new orb

    def _calculate_reward(self):
        reward = 0

        # Reward for filling a slot
        if any(s['filled_this_step'] for s in self.slots):
            reward += 10.0
            self.momentum = min(100.0, self.momentum + 25)

        # Find closest matching unfilled slot for continuous reward
        if self.orb:
            target_slot = None
            min_dist = float("inf")
            for slot in self.slots:
                if not slot["filled"] and slot["color_name"] == self.orb["color_name"]:
                    dist = self.orb["pos"].distance_to(slot["pos"])
                    if dist < min_dist:
                        min_dist = dist
                        target_slot = slot

            if target_slot:
                if self.orb_state == "moving":
                    dist_change = self.prev_orb_dist_to_target - min_dist
                    reward += dist_change * 0.005  # Small reward for getting closer
                self.prev_orb_dist_to_target = min_dist

        return reward

    def _check_termination(self):
        win = all(s["filled"] for s in self.slots)
        lose = self.momentum <= 0

        if win:
            self.score += 100
            self.game_over = True
            # sfx: win_game.wav
        elif lose:
            self.score -= 100
            self.game_over = True
            # sfx: lose_game.wav

        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "momentum": self.momentum}

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw obstacles
        for obs_rect in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_rect)

        # Draw slots
        for slot in self.slots:
            pos = (int(slot["pos"].x), int(slot["pos"].y))
            if slot["filled"]:
                self._render_glow(self.screen, pos, slot["radius"], slot["color"], steps=5)
                pygame.gfxdraw.filled_circle(
                    self.screen, pos[0], pos[1], slot["radius"], slot["color"]
                )
            else:
                pygame.gfxdraw.aacircle(
                    self.screen, pos[0], pos[1], slot["radius"], slot["color"]
                )
                pygame.gfxdraw.aacircle(
                    self.screen, pos[0], pos[1], slot["radius"] - 1, slot["color"]
                )

        # Draw particles
        for p in self.particles:
            p_pos = (int(p["pos"].x), int(p["pos"].y))
            alpha = int(255 * (p["life"] / p["max_life"]))
            try:
                # Use a temporary surface for blending
                temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, int(p['size']), int(p['size']), int(p['size']), p['color'] + (alpha,))
                self.screen.blit(temp_surf, (p_pos[0] - int(p['size']), p_pos[1] - int(p['size'])))
            except (ValueError, TypeError): # Handle color format issues
                pass


        # Draw orb and aim vector
        if self.orb:
            pos = (int(self.orb["pos"].x), int(self.orb["pos"].y))
            self._render_glow(self.screen, pos, self.orb["radius"], self.orb["color"])
            pygame.gfxdraw.filled_circle(
                self.screen, pos[0], pos[1], self.orb["radius"], self.orb["color"]
            )
            pygame.gfxdraw.aacircle(
                self.screen, pos[0], pos[1], self.orb["radius"], (255, 255, 255)
            )

            if self.orb_state == "aiming" and self.aim_vector.length() > 0:
                start_pos = self.orb["pos"]
                end_pos = self.orb["pos"] + self.aim_vector
                pygame.draw.aaline(
                    self.screen, (255, 255, 255, 150), start_pos, end_pos, 2
                )
                pygame.gfxdraw.aacircle(
                    self.screen, int(end_pos.x), int(end_pos.y), 5, (255, 255, 255, 200)
                )

    def _render_ui(self):
        # Momentum bar
        bar_width = self.WIDTH - 20
        momentum_frac = max(0, self.momentum / 100.0)
        pygame.draw.rect(
            self.screen, self.COLOR_MOMENTUM_BAR_BG, (10, 10, bar_width, 20)
        )
        pygame.draw.rect(
            self.screen,
            self.COLOR_MOMENTUM_BAR,
            (10, 10, int(bar_width * momentum_frac), 20),
        )

        # Score text
        score_text = self.font_large.render(
            f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT
        )
        self.screen.blit(score_text, (10, self.HEIGHT - 30))

        # Gravity indicator
        center = (self.WIDTH - 40, self.HEIGHT - 40)
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], 25, self.COLOR_GRAVITY)
        end_pos = (
            center[0] + self.gravity.x * 150,
            center[1] + self.gravity.y * 150,
        )
        pygame.draw.line(self.screen, self.COLOR_GRAVITY, center, end_pos, 2)

    def _render_glow(self, surface, pos, radius, color, steps=10):
        for i in range(steps, 0, -1):
            alpha = int(50 * (1 - (i / steps)))
            glow_color = color + (alpha,)
            try:
                # Use a temporary surface for blending
                temp_surf = pygame.Surface(( (radius+i)*2, (radius+i)*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, radius+i, radius+i, radius + i, glow_color)
                surface.blit(temp_surf, (pos[0] - (radius+i), pos[1] - (radius+i)))
            except (ValueError, TypeError): # Handle color format issues
                pass

    def _generate_level(self):
        self.slots.clear()
        self.obstacles.clear()
        num_slots = self.np_random.integers(3, 6)
        available_colors = list(GameEnv.UNLOCKED_COLORS)

        for i in range(num_slots):
            color_name = self.np_random.choice(available_colors)
            self.slots.append(
                {
                    "pos": pygame.Vector2(
                        self.np_random.integers(50, self.WIDTH - 50),
                        self.np_random.integers(50, self.HEIGHT - 50),
                    ),
                    "color_name": color_name,
                    "color": self.COLOR_MAP[color_name],
                    "radius": 15,
                    "filled": False,
                    "filled_this_step": False,  # Helper for reward calculation
                }
            )

        for _ in range(3):  # Start with 3 obstacles
            self._toggle_obstacle(force_add=True)

    def _spawn_orb(self):
        unfilled_slots = [s for s in self.slots if not s["filled"]]
        if not unfilled_slots:
            self.orb = None
            self.game_over = True # Win condition
            return

        target_slot = self.np_random.choice(unfilled_slots)
        self.orb = {
            "pos": pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2),
            "vel": pygame.Vector2(0, 0),
            "color_name": target_slot["color_name"],
            "color": target_slot["color"],
            "radius": 10,
        }
        self.orb_state = "aiming"
        self.prev_orb_dist_to_target = self.orb["pos"].distance_to(target_slot["pos"])

    def _toggle_obstacle(self, force_add=False):
        if (
            len(self.obstacles) > 0
            and not force_add
            and self.np_random.random() < 0.4
        ):
            self.obstacles.pop(self.np_random.integers(len(self.obstacles)))
        elif len(self.obstacles) < 5:
            w, h = self.np_random.integers(40, 120), self.np_random.integers(10, 20)
            if self.np_random.random() < 0.5:
                w, h = h, w  # Random orientation
            x = self.np_random.integers(80, self.WIDTH - 80 - w)
            y = self.np_random.integers(80, self.HEIGHT - 80 - h)
            self.obstacles.append(pygame.Rect(x, y, w, h))

    def _create_particles(self, pos, color):
        for _ in range(30):
            self.particles.append(
                {
                    "pos": pos.copy(),
                    "vel": pygame.Vector2(
                        self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)
                    ).normalize()
                    * self.np_random.uniform(1, 4),
                    "life": self.np_random.integers(20, 40),
                    "max_life": 40,
                    "color": color,
                    "size": self.np_random.uniform(1, 4),
                }
            )

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_unlocks(self):
        if GameEnv.TOTAL_SLOTS_FILLED_EVER >= 5:
            GameEnv.UNLOCKED_COLORS.add("purple")
        if GameEnv.TOTAL_SLOTS_FILLED_EVER >= 10:
            GameEnv.UNLOCKED_COLORS.add("orange")

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # --- Manual Play Testing ---
    # To run this, you need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Orb Flipper")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement, space, shift = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(
                f"Episode Finished! Final Score: {info['score']:.2f}, Steps: {info['steps']}"
            )
            obs, info = env.reset()
            total_reward = 0

        # Blit the observation from the environment to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()