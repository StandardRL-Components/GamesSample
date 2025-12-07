
# Generated: 2025-08-27T20:06:20.411460
# Source Brief: brief_02346.md
# Brief Index: 2346

        
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
        "Controls: Use ←→ for air control. Hold [SPACE] on a platform to "
        "charge a jump, then release to leap. Reach the green platform at the top!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist arcade platformer. Charge your jumps and hop between "
        "platforms to reach the summit. The higher you are, the more points you score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000

        # Colors
        self.COLOR_BG_TOP = (50, 80, 150)
        self.COLOR_BG_BOTTOM = (10, 20, 50)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLATFORM = (240, 240, 240)
        self.COLOR_GOAL = (50, 255, 50)
        self.COLOR_PARTICLE = (255, 255, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_CHARGE_BAR = (100, 200, 255)
        self.COLOR_CHARGE_BAR_BG = (50, 50, 50, 150)

        # Physics & Gameplay
        self.GRAVITY = 0.6
        self.AIR_CONTROL_FORCE = 0.5
        self.MAX_JUMP_CHARGE = 30  # frames
        self.JUMP_POWER_BASE = 10.0
        self.JUMP_POWER_SCALAR = 0.25
        self.MAX_VEL_Y = 15
        self.MAX_VEL_X = 5
        self.FRICTION = 0.95
        self.PLAYER_SIZE = (16, 16)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("monospace", 24, bold=True)
            self.small_font = pygame.font.SysFont("monospace", 16)
        except pygame.error:
            self.font = pygame.font.Font(None, 30)
            self.small_font = pygame.font.Font(None, 20)
        
        self._bg_surface = self._create_gradient_background()
        
        # Initialize state variables
        self.reset()

    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = tuple(
                int(self.COLOR_BG_BOTTOM[i] + interp * (self.COLOR_BG_TOP[i] - self.COLOR_BG_BOTTOM[i]))
                for i in range(3)
            )
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def _generate_platforms(self):
        self.platforms = []
        
        start_plat = pygame.Rect(self.WIDTH // 2 - 50, self.HEIGHT - 40, 100, 15)
        self.platforms.append(start_plat)
        self.start_y = float(start_plat.top)

        self.goal_platform = pygame.Rect(self.WIDTH // 2 - 50, 30, 100, 15)

        current_y = self.HEIGHT - 120
        last_x = float(start_plat.centerx)
        
        while current_y > 100:
            max_reach = 180
            x_pos = self.np_random.uniform(last_x - max_reach, last_x + max_reach)
            width = self.np_random.integers(60, 120)
            x_pos = np.clip(x_pos - width / 2, 20, self.WIDTH - width - 20)
            
            plat = pygame.Rect(int(x_pos), int(current_y), int(width), 10)
            self.platforms.append(plat)
            
            last_x = float(plat.centerx)
            current_y -= self.np_random.integers(50, 85)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_platforms()

        start_platform = self.platforms[0]
        self.player_pos = [float(start_platform.centerx), float(start_platform.top - self.PLAYER_SIZE[1])]
        self.player_vel = [0.0, 0.0]
        self.player_rect = pygame.Rect(0, 0, *self.PLAYER_SIZE)
        self.on_ground = True
        self.jump_charge = 0
        self.prev_space_held = False
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            if self.auto_advance: self.clock.tick(self.FPS)
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Player Input & Actions ---
        if self.on_ground and space_held:
            self.jump_charge = min(self.MAX_JUMP_CHARGE, self.jump_charge + 1)
        
        if self.on_ground and self.prev_space_held and not space_held:
            jump_force = self.JUMP_POWER_BASE + self.jump_charge * self.JUMP_POWER_SCALAR
            self.player_vel[1] = -jump_force
            self.on_ground = False
            self.jump_charge = 0
            # sfx: jump

        self.prev_space_held = space_held

        # --- Physics & Movement ---
        if not self.on_ground:
            self.player_vel[1] += self.GRAVITY
            if movement == 3: self.player_vel[0] -= self.AIR_CONTROL_FORCE
            elif movement == 4: self.player_vel[0] += self.AIR_CONTROL_FORCE
        
        self.player_vel[0] *= self.FRICTION
        self.player_vel[0] = np.clip(self.player_vel[0], -self.MAX_VEL_X, self.MAX_VEL_X)
        self.player_vel[1] = np.clip(self.player_vel[1], -float('inf'), self.MAX_VEL_Y)

        prev_player_rect = self.player_rect.copy()
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]
        self.player_rect.topleft = (int(self.player_pos[0]), int(self.player_pos[1]))

        # --- Collisions ---
        if self.player_rect.left < 0:
            self.player_rect.left = 0
            self.player_pos[0] = 0.0
            self.player_vel[0] *= -0.5
        if self.player_rect.right > self.WIDTH:
            self.player_rect.right = self.WIDTH
            self.player_pos[0] = float(self.WIDTH - self.PLAYER_SIZE[0])
            self.player_vel[0] *= -0.5

        landed_on_plat = None
        if self.player_vel[1] >= 0:
            all_plats = self.platforms + [self.goal_platform]
            for plat in all_plats:
                if self.player_rect.colliderect(plat) and prev_player_rect.bottom <= plat.top:
                    landed_on_plat = plat
                    break
        
        if landed_on_plat and not self.on_ground:
            self.on_ground = True
            self.player_pos[1] = float(landed_on_plat.top - self.PLAYER_SIZE[1])
            self.player_rect.bottom = landed_on_plat.top
            self.player_vel = [0.0, 0.0]
            reward += 1.0
            self._create_particles(self.player_rect.midbottom, 15)
            # sfx: land
        elif not landed_on_plat:
            self.on_ground = False

        self._update_particles()

        # --- Rewards & Termination ---
        if self.player_pos[1] < self.start_y:
            reward += 0.01 * ((self.start_y - self.player_pos[1]) / self.HEIGHT)
        else:
            reward -= 0.01

        terminated = False
        if self.player_rect.top > self.HEIGHT:
            terminated = True
            reward -= 10.0
        
        if self.player_rect.colliderect(self.goal_platform) and self.on_ground:
            terminated = True
            reward += 100.0
            self._create_particles(self.player_rect.center, 50, particle_color=(50, 255, 50))
            # sfx: win

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward

        if self.auto_advance: self.clock.tick(self.FPS)

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _create_particles(self, pos, count, particle_color=None):
        if particle_color is None:
            particle_color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.uniform(math.pi, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 5)
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': list(pos), 'vel': vel, 'radius': radius, 'life': life, 'color': particle_color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2
            p['life'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        self.screen.blit(self._bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)
            pygame.draw.rect(self.screen, tuple(int(c * 0.8) for c in self.COLOR_PLATFORM), plat.inflate(-4, -4), border_radius=3)

        pygame.draw.rect(self.screen, self.COLOR_GOAL, self.goal_platform, border_radius=3)
        pygame.draw.rect(self.screen, tuple(int(c * 0.8) for c in self.COLOR_GOAL), self.goal_platform.inflate(-4, -4), border_radius=3)

        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 15))))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=2)
        inner_rect = self.player_rect.inflate(-4, -4)
        pygame.draw.rect(self.screen, tuple(min(255, c + 50) for c in self.COLOR_PLAYER), inner_rect, border_radius=2)

        if self.jump_charge > 0:
            charge_ratio = self.jump_charge / self.MAX_JUMP_CHARGE
            bar_width, bar_height = self.PLAYER_SIZE[0] * 1.5, 5
            bar_x = self.player_rect.centerx - bar_width / 2
            bar_y = self.player_rect.bottom + 5
            
            bg_surf = pygame.Surface((bar_width, bar_height), pygame.SRCALPHA)
            bg_surf.fill(self.COLOR_CHARGE_BAR_BG)
            self.screen.blit(bg_surf, (int(bar_x), int(bar_y)))

            fill_width = bar_width * charge_ratio
            pygame.draw.rect(self.screen, self.COLOR_CHARGE_BAR, (int(bar_x), int(bar_y), int(fill_width), bar_height), border_radius=2)

    def _render_ui(self):
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        steps_surf = self.small_font.render(steps_text, True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (self.WIDTH - steps_surf.get_width() - 10, 10))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and obs.dtype == np.uint8
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and obs.dtype == np.uint8
        assert isinstance(reward, float) and isinstance(term, bool) and not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import os
    # To run headlessly (e.g., on a server), you can use a dummy video driver
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    # To run with a window for human play
    render_mode = "human"
    if render_mode == "human":
        # Monkey-patch the environment for human rendering
        GameEnv.metadata["render_modes"].append("human")
        _original_init = GameEnv.__init__
        def new_init(self, render_mode="human", **kwargs):
            _original_init(self, render_mode=render_mode, **kwargs)
            self.render_mode = render_mode
            if self.render_mode == "human":
                self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
                pygame.display.set_caption("Platform Hopper")
        GameEnv.__init__ = new_init

        _original_get_obs = GameEnv._get_observation
        def new_get_obs(self):
            obs = _original_get_obs(self)
            if self.render_mode == "human":
                self.window.blit(self.screen, (0, 0))
                pygame.display.flip()
            return obs
        GameEnv._get_observation = new_get_obs

    env = GameEnv(render_mode=render_mode)
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    print("\n--- Starting Human Play Test ---")
    print(env.user_guide)

    while not done:
        # Simple human controls
        keys = pygame.key.get_pressed()
        mov = 0 # none
        if keys[pygame.K_UP]: mov = 1
        if keys[pygame.K_DOWN]: mov = 2
        if keys[pygame.K_LEFT]: mov = 3
        if keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        if done and not (event.type == pygame.QUIT if 'event' in locals() else False):
            print(f"Episode finished! Final Score: {info['score']:.2f}. Resetting...")
            obs, info = env.reset()
            done = False

    env.close()
    print("\nEnvironment test complete.")