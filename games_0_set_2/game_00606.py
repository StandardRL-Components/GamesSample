
# Generated: 2025-08-27T14:11:54.046892
# Source Brief: brief_00606.md
# Brief Index: 606

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑/↓ to aim launcher. Space to launch ball. Break all the blocks!"
    )

    game_description = (
        "A classic block-breaker arcade game. Aim carefully to clear the board, create combos, and collect powerful multi-ball power-ups."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_LAUNCHER = (200, 200, 220)
        self.COLOR_TEXT = (230, 230, 230)
        self.BLOCK_COLORS = [
            (66, 135, 245), (66, 245, 135), (245, 245, 66), (245, 135, 66), (245, 66, 135)
        ]
        self.POWERUP_COLOR = (180, 100, 255)

        # Game parameters
        self.BALL_RADIUS = 6
        self.BALL_SPEED = 8
        self.LAUNCHER_POS = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT - 20)
        self.MAX_BALLS = 3
        self.BLOCK_ROWS = 5
        self.BLOCK_COLS = 10
        self.BLOCK_WIDTH = 58
        self.BLOCK_HEIGHT = 18
        self.POWERUP_CHANCE = 0.15

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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_phase = "aiming"
        self.launcher_angle = 0.0
        self.balls_left = 0
        self.balls = []
        self.blocks = []
        self.particles = []
        self.powerups = []
        self.combo = 0
        self.space_was_pressed = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_phase = "aiming"
        self.launcher_angle = -math.pi / 2
        self.balls_left = self.MAX_BALLS
        self.balls.clear()
        self.particles.clear()
        self.powerups.clear()
        self.combo = 0
        self.space_was_pressed = False

        self._create_blocks()

        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        self.blocks.clear()
        total_block_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + 2)
        start_x = (self.WIDTH - total_block_width) / 2
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                x = start_x + j * (self.BLOCK_WIDTH + 2)
                y = 40 + i * (self.BLOCK_HEIGHT + 2)
                rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                points = (self.BLOCK_ROWS - i) * 10
                self.blocks.append({"rect": rect, "color": color, "points": points})

    def step(self, action):
        reward = 0
        
        self._handle_input(action)
        
        step_reward = self._update_game_state()
        reward += step_reward

        self._update_effects()

        terminated = self._check_termination()
        if terminated:
            if not self.blocks:
                reward += 100
            elif self.balls_left == 0 and not self.balls:
                reward -= 100

        self.steps += 1
        self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_pressed, _ = action
        space_pressed = space_pressed == 1

        if self.game_phase == "aiming":
            angle_change = 0.05
            if movement == 1:
                self.launcher_angle -= angle_change
            elif movement == 2:
                self.launcher_angle += angle_change

            self.launcher_angle = np.clip(self.launcher_angle, -math.pi + 0.1, -0.1)

            if space_pressed and not self.space_was_pressed:
                self._launch_ball()
        
        self.space_was_pressed = space_pressed

    def _launch_ball(self):
        if self.balls_left > 0:
            self.balls_left -= 1
            self.game_phase = "playing"
            self.combo = 0
            
            vel = pygame.math.Vector2(
                math.cos(self.launcher_angle), math.sin(self.launcher_angle)
            ) * self.BALL_SPEED
            
            ball = {
                "pos": self.LAUNCHER_POS.copy(),
                "vel": vel,
                "radius": self.BALL_RADIUS,
            }
            self.balls.append(ball)
            # sfx: launch_sound.play()

    def _update_game_state(self):
        step_reward = 0
        if self.game_phase != "playing":
            return step_reward

        balls_to_remove = []
        for i, ball in enumerate(self.balls):
            ball["pos"] += ball["vel"]

            # Wall collisions
            if ball["pos"].x <= ball["radius"] or ball["pos"].x >= self.WIDTH - ball["radius"]:
                ball["vel"].x *= -1
                ball["pos"].x = np.clip(ball["pos"].x, ball["radius"], self.WIDTH - ball["radius"])
                # sfx: wall_bounce.play()
            if ball["pos"].y <= ball["radius"]:
                ball["vel"].y *= -1
                ball["pos"].y = np.clip(ball["pos"].y, ball["radius"], self.HEIGHT)
                # sfx: wall_bounce.play()

            # Ball lost
            if ball["pos"].y >= self.HEIGHT + ball["radius"] * 2:
                balls_to_remove.append(i)
                continue

            # Block collisions
            for block in reversed(self.blocks):
                if self._check_ball_block_collision(ball, block):
                    step_reward += 1
                    self.score += block["points"] + self.combo
                    self.combo += 1
                    self._create_particles(block["rect"].center, block["color"], 20)
                    if random.random() < self.POWERUP_CHANCE:
                        self._spawn_powerup(block["rect"].center)
                    self.blocks.remove(block)
                    # sfx: block_break.play()
                    break 
            
            # Powerup collisions
            for powerup in reversed(self.powerups):
                if powerup['pos'].distance_to(ball['pos']) < powerup['radius'] + ball['radius']:
                    step_reward += 5
                    self.score += 50
                    self._apply_powerup(powerup['type'])
                    self.powerups.remove(powerup)
                    # sfx: powerup_collect.play()

        for i in sorted(balls_to_remove, reverse=True):
            del self.balls[i]
            # sfx: ball_lost.play()

        if not self.balls:
            self.game_phase = "aiming"
            self.combo = 0

        return step_reward

    def _check_ball_block_collision(self, ball, block):
        closest_x = np.clip(ball["pos"].x, block["rect"].left, block["rect"].right)
        closest_y = np.clip(ball["pos"].y, block["rect"].top, block["rect"].bottom)

        dist_vec = ball["pos"] - pygame.math.Vector2(closest_x, closest_y)
        
        if dist_vec.length_squared() < ball["radius"] ** 2:
            overlap_rect = block["rect"].clip(
                pygame.Rect(
                    ball["pos"].x - ball["radius"], ball["pos"].y - ball["radius"],
                    ball["radius"]*2, ball["radius"]*2
                )
            )
            
            if overlap_rect.width < overlap_rect.height:
                ball["vel"].x *= -1
                if ball["pos"].x < block["rect"].centerx:
                    ball["pos"].x = block["rect"].left - ball["radius"]
                else:
                    ball["pos"].x = block["rect"].right + ball["radius"]
            else:
                ball["vel"].y *= -1
                if ball["pos"].y < block["rect"].centery:
                    ball["pos"].y = block["rect"].top - ball["radius"]
                else:
                    ball["pos"].y = block["rect"].bottom + ball["radius"]
            return True
        return False

    def _spawn_powerup(self, position):
        self.powerups.append({
            'pos': pygame.math.Vector2(position),
            'type': 'multi_ball',
            'radius': 10,
            'spawn_time': self.steps
        })

    def _apply_powerup(self, powerup_type):
        if powerup_type == 'multi_ball' and self.balls:
            original_ball = self.balls[0]
            for angle_offset in [-0.3, 0.3]:
                new_vel = original_ball['vel'].rotate_rad(angle_offset)
                new_ball = {
                    "pos": original_ball['pos'].copy(),
                    "vel": new_vel,
                    "radius": self.BALL_RADIUS,
                }
                self.balls.append(new_ball)

    def _update_effects(self):
        for p in reversed(self.particles):
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        win = not self.blocks
        loss = self.balls_left == 0 and not self.balls
        timeout = self.steps >= self.MAX_STEPS
        return win or loss or timeout

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = random.randint(10, 20)
            self.particles.append({"pos": pygame.math.Vector2(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_blocks()
        self._render_powerups()
        self._render_particles()
        self._render_launcher()
        self._render_balls()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_blocks(self):
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            highlight_color = tuple(min(255, c + 30) for c in block["color"])
            pygame.draw.rect(self.screen, highlight_color, (block["rect"].x+2, block["rect"].y+2, block["rect"].width-4, 5), border_radius=2)

    def _render_launcher(self):
        end_pos = self.LAUNCHER_POS + pygame.math.Vector2(math.cos(self.launcher_angle), math.sin(self.launcher_angle)) * 30
        pygame.draw.line(self.screen, self.COLOR_LAUNCHER, self.LAUNCHER_POS, end_pos, 5)
        
        if self.game_phase == "aiming":
            for i in range(1, 20):
                p1 = self.LAUNCHER_POS + pygame.math.Vector2(math.cos(self.launcher_angle), math.sin(self.launcher_angle)) * (i * 20)
                p2 = self.LAUNCHER_POS + pygame.math.Vector2(math.cos(self.launcher_angle), math.sin(self.launcher_angle)) * (i * 20 + 10)
                if p1.y < 0 or p2.y < 0: break
                pygame.draw.line(self.screen, self.COLOR_LAUNCHER, p1, p2, 1)

    def _render_balls(self):
        for ball in self.balls:
            pos = (int(ball["pos"].x), int(ball["pos"].y))
            radius = int(ball["radius"])
            glow_color = (150, 150, 150)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 3, glow_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 3, glow_color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_BALL)

    def _render_powerups(self):
        for p in self.powerups:
            pulse = abs(math.sin((self.steps - p['spawn_time']) * 0.1))
            radius = int(p['radius'] * (1 + pulse * 0.2))
            alpha = int(150 + pulse * 105)
            
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            color = self.POWERUP_COLOR + (alpha,)
            pygame.gfxdraw.filled_circle(s, radius, radius, radius, color)
            pygame.gfxdraw.aacircle(s, radius, radius, radius, color)
            self.screen.blit(s, (int(p['pos'].x - radius), int(p['pos'].y - radius)))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, int(255 * (p["lifespan"] / 20)))
            color = p["color"] + (alpha,)
            size = max(1, int(p["lifespan"] / 5))
            
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (size, size), size)
            self.screen.blit(s, (int(p['pos'].x - size), int(p['pos'].y - size)))

    def _render_ui(self):
        score_surf = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        balls_text_surf = self.font_small.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(balls_text_surf, (self.WIDTH - 150, 10))
        for i in range(self.balls_left):
            pos = (self.WIDTH - 80 + i * 20, 18)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 6, self.COLOR_BALL)
        
        if self.combo > 2:
            combo_surf = self.font_large.render(f"x{self.combo}", True, self.COLOR_TEXT)
            combo_rect = combo_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 100))
            self.screen.blit(combo_surf, combo_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
            "game_phase": self.game_phase
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker Gym Environment")
    
    terminated = False
    
    while not terminated:
        action = np.array([0, 0, 0])
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

    print(f"Game Over! Final Score: {info['score']}")
    env.close()