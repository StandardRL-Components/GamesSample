
# Generated: 2025-08-27T23:37:51.044470
# Source Brief: brief_03532.md
# Brief Index: 3532

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to swim, hold Space to boost. Avoid the red predators and reach the surface!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape from the deep ocean by dodging predators and reaching the surface. Use your boost wisely to get out of tight spots."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    COLOR_BG_TOP = (10, 80, 150)
    COLOR_BG_BOTTOM = (0, 10, 40)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 150)
    COLOR_PREDATOR = (220, 20, 20)
    COLOR_PREDATOR_FIN = (180, 20, 20)
    COLOR_SURFACE = (200, 220, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_BOOST_BAR = (0, 200, 255)
    COLOR_BOOST_BAR_BG = (50, 50, 80)
    MAX_STEPS = 10000
    NUM_PREDATORS = 3
    PLAYER_BASE_SPEED = 2.5
    PLAYER_BOOST_MULTIPLIER = 2.0
    BOOST_DEPLETION_RATE = 2.0
    BOOST_RECHARGE_RATE = 0.5
    PREDATOR_SPEED_INCREASE_INTERVAL = 500
    PREDATOR_SPEED_INCREASE_AMOUNT = 0.1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 20)
        except pygame.error:
            self.font_ui = pygame.font.SysFont("sans", 20)
        
        # State variables initialized in reset
        self.player_pos = None
        self.player_radius = 10
        self.predators = None
        self.particles = None
        self.boost_meter = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.np_random = None
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT - 50.0])
        self.boost_meter = 100.0
        
        self.predators = []
        for i in range(self.NUM_PREDATORS):
            size = self.np_random.integers(15, 26)
            self.predators.append({
                "pos": np.array([
                    self.np_random.uniform(0, self.SCREEN_WIDTH),
                    self.np_random.uniform(50, self.SCREEN_HEIGHT - 100)
                ]),
                "radius": size,
                "base_speed": self.np_random.uniform(1.0, 2.0),
                "direction": self.np_random.choice([-1, 1]),
                "path": {
                    "amplitude": self.np_random.uniform(20, 60),
                    "frequency": self.np_random.uniform(0.005, 0.015),
                    "phase": self.np_random.uniform(0, 2 * math.pi),
                    "base_y": self.np_random.uniform(size, self.SCREEN_HEIGHT - size - 100)
                }
            })

        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Player Movement & Boost ---
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] = -1
        elif movement == 2: move_vec[1] = 1
        elif movement == 3: move_vec[0] = -1
        elif movement == 4: move_vec[0] = 1
        
        if np.linalg.norm(move_vec) > 0:
            move_vec /= np.linalg.norm(move_vec)

        is_boosting = space_held and self.boost_meter > 0
        speed = self.PLAYER_BASE_SPEED * (self.PLAYER_BOOST_MULTIPLIER if is_boosting else 1.0)
        
        if is_boosting:
            self.boost_meter = max(0, self.boost_meter - self.BOOST_DEPLETION_RATE)
            # Sound: boost_sound.play()
            if self.steps % 2 == 0: self._create_boost_particles()
        else:
            self.boost_meter = min(100, self.boost_meter + self.BOOST_RECHARGE_RATE)
        
        self.player_pos += move_vec * speed
        self._update_player_bounds()

        # --- Game Logic Update ---
        self._update_predators()
        self._update_particles()
        
        # --- Termination & Reward ---
        terminated = False
        reward = 0.1  # Survival reward

        if is_boosting: reward -= 0.2

        if self._check_collision():
            terminated = True
            # Sound: player_hit.play()
        elif self.player_pos[1] <= self.player_radius:
            terminated = True
            reward += 10.0 # Win bonus
            # Sound: win_jingle.play()
        elif self.steps >= self.MAX_STEPS - 1:
            terminated = True

        if terminated: self.game_over = True
        
        self.score += reward
        self.steps += 1
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player_bounds(self):
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_radius, self.SCREEN_WIDTH - self.player_radius)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.SCREEN_HEIGHT - self.player_radius)

    def _update_predators(self):
        if self.steps > 0 and self.steps % self.PREDATOR_SPEED_INCREASE_INTERVAL == 0:
            for p in self.predators:
                p["base_speed"] += self.PREDATOR_SPEED_INCREASE_AMOUNT
        
        for p in self.predators:
            p["pos"][0] += p["base_speed"] * p["direction"]
            p["pos"][1] = p["path"]["base_y"] + p["path"]["amplitude"] * math.sin(p["path"]["frequency"] * p["pos"][0] + p["path"]["phase"])

            if p["pos"][0] > self.SCREEN_WIDTH + p["radius"]: p["pos"][0] = -p["radius"]
            elif p["pos"][0] < -p["radius"]: p["pos"][0] = self.SCREEN_WIDTH + p["radius"]

    def _create_boost_particles(self):
        for _ in range(3):
            self.particles.append({
                "pos": self.player_pos.copy() + self.np_random.uniform(-5, 5, size=2),
                "vel": self.np_random.uniform(-1, 1, size=2),
                "radius": self.np_random.uniform(2, 5),
                "lifetime": self.np_random.integers(15, 30)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["radius"] *= 0.95
            p["lifetime"] -= 1
        self.particles = [p for p in self.particles if p["lifetime"] > 0 and p["radius"] > 0.5]

    def _check_collision(self):
        for predator in self.predators:
            dist_sq = np.sum((self.player_pos - predator["pos"])**2)
            if dist_sq < (self.player_radius + predator["radius"])**2:
                return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "depth": max(0, self.SCREEN_HEIGHT - self.player_pos[1]),
            "boost": self.boost_meter,
        }

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = tuple(int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp) for i in range(3))
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_SURFACE, (0, 0, self.SCREEN_WIDTH, 5))
        
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / 30))
            color = (*self.COLOR_SURFACE, alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            radius = int(p["radius"])
            if radius > 0:
                pygame.gfxdraw.aacircle(self.screen, *pos, radius, color)
                pygame.gfxdraw.filled_circle(self.screen, *pos, radius, color)

        for p in self.predators: self._draw_predator(p)
        self._draw_player()

    def _draw_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        r = int(self.player_radius)
        
        glow_radius = int(r * 1.8)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.COLOR_PLAYER_GLOW, 80), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        points = [(x, y - r), (x - r // 2, y + r // 2), (x + r // 2, y + r // 2)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _draw_predator(self, predator):
        x, y = int(predator["pos"][0]), int(predator["pos"][1])
        r = int(predator["radius"])
        direction = predator["direction"]

        body_points = [
            (x + r * direction, y), (x - r * direction, y - r * 0.6),
            (x - r * 0.7 * direction, y), (x - r * direction, y + r * 0.6)
        ]
        pygame.gfxdraw.aapolygon(self.screen, body_points, self.COLOR_PREDATOR)
        pygame.gfxdraw.filled_polygon(self.screen, body_points, self.COLOR_PREDATOR)

        fin_angle = math.sin(self.steps * 0.3 + predator["path"]["phase"]) * 0.5
        fin_base_x = x - r * 0.8 * direction
        fin_tip_x = fin_base_x - math.cos(fin_angle) * r * 0.8 * direction
        fin_tip_y = y + math.sin(fin_angle) * r * 0.8
        
        fin_points = [(x - r * 0.6 * direction, y), (fin_tip_x, fin_tip_y), (x - r * direction, y)]
        pygame.gfxdraw.aapolygon(self.screen, fin_points, self.COLOR_PREDATOR_FIN)
        pygame.gfxdraw.filled_polygon(self.screen, fin_points, self.COLOR_PREDATOR_FIN)

    def _render_ui(self):
        depth = max(0, self.SCREEN_HEIGHT - self.player_pos[1])
        depth_text = self.font_ui.render(f"DEPTH: {int(depth)}m", True, self.COLOR_TEXT)
        self.screen.blit(depth_text, (10, 10))

        time_text = self.font_ui.render(f"TIME: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10)))
        
        bar_w, bar_h, bar_x, bar_y = 150, 15, (self.SCREEN_WIDTH - 150) / 2, self.SCREEN_HEIGHT - 25
        pygame.draw.rect(self.screen, self.COLOR_BOOST_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        fill_w = max(0, (self.boost_meter / 100) * bar_w)
        pygame.draw.rect(self.screen, self.COLOR_BOOST_BAR, (bar_x, bar_y, fill_w, bar_h), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_w, bar_h), 1, border_radius=3)

    def close(self):
        pygame.quit()

# Example usage and validation block
if __name__ == '__main__':
    def validate_implementation(env_instance):
        print("Running implementation validation...")
        assert env_instance.action_space.shape == (3,)
        assert env_instance.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = env_instance._get_observation()
        assert test_obs.shape == (GameEnv.SCREEN_HEIGHT, GameEnv.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = env_instance.reset()
        assert obs.shape == (GameEnv.SCREEN_HEIGHT, GameEnv.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = env_instance.action_space.sample()
        obs, reward, term, trunc, info = env_instance.step(test_action)
        assert obs.shape == (GameEnv.SCREEN_HEIGHT, GameEnv.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = GameEnv()
    validate_implementation(env)
    env.close()

    # To visualize the game, comment out the os.environ line above and run this block
    # from gymnasium.utils.play import play
    # play(GameEnv(), keys_to_action={
    #     "w": np.array([1, 0, 0]), "s": np.array([2, 0, 0]),
    #     "a": np.array([3, 0, 0]), "d": np.array([4, 0, 0]),
    #     " ": np.array([0, 1, 0]),
    # }, noop=np.array([0, 0, 0]), fps=30)