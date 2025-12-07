import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:47:54.005345
# Source Brief: brief_00692.md
# Brief Index: 692
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
        "Time your shots to launch pegs into the moving slots of two rotating gears. "
        "Adjust gear speeds to align the slots and complete the puzzle before time runs out."
    )
    user_guide = (
        "Use ↑/↓ to control the left gear's speed and ←/→ for the right gear. Press space to launch a peg."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME = 60.0
        self.MAX_STEPS = int(self.MAX_TIME * self.FPS)

        # Colors
        self.COLOR_BG = (15, 20, 45)
        self.COLOR_GEAR = (80, 90, 110)
        self.COLOR_GEAR_OUTLINE = (120, 130, 150)
        self.COLOR_SLOT_SUCCESS = (60, 255, 120)
        self.COLOR_PEG = (255, 80, 80)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_LAUNCHER = (180, 180, 200)

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
            self.font_small = pygame.font.SysFont("Consolas", 20)
            self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 52)


        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.timer = None
        self.pegs_to_insert_total = None
        self.pegs_inserted_count = None
        self.gears = None
        self.pegs = None
        self.particles = None
        self.time_since_direction_reversal = None
        self.prev_space_held = None
        self.background_stars = None

        # Initialize state for validation check
        self._initialize_state()
        self.validate_implementation()

    def _initialize_state(self):
        """Initializes a minimal state for validation before the first reset."""
        self.steps = 0
        self.score = 0
        self.game_over = True
        self.timer = self.MAX_TIME
        self.pegs_to_insert_total = 1
        self.pegs_inserted_count = 0
        self.gears = []
        self.pegs = []
        self.particles = []
        self.time_since_direction_reversal = 0.0
        self.prev_space_held = False
        self.background_stars = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME
        self.pegs_to_insert_total = 1
        self.pegs_inserted_count = 0
        self.pegs = []
        self.particles = []
        self.prev_space_held = False
        self.time_since_direction_reversal = 0.0

        gear_radius = 60
        gear_y = self.HEIGHT // 2 - 40
        self.gears = [
            {
                "center": np.array([self.WIDTH * 0.3, gear_y], dtype=float),
                "radius": gear_radius,
                "angle": self.np_random.uniform(0, 360),
                "angular_velocity_rps": 1.5,
                "direction": 1,
                "slots": [0, 120, 240]
            },
            {
                "center": np.array([self.WIDTH * 0.7, gear_y], dtype=float),
                "radius": gear_radius,
                "angle": self.np_random.uniform(0, 360),
                "angular_velocity_rps": 1.5,
                "direction": -1,
                "slots": [60, 180, 300]
            }
        ]

        # Generate stars using the seeded RNG
        xs = self.np_random.integers(0, self.WIDTH, size=100)
        ys = self.np_random.integers(0, self.HEIGHT, size=100)
        sizes = self.np_random.integers(1, 3, size=100) # upper bound is exclusive, so 1 or 2
        self.background_stars = list(zip(xs, ys, sizes))

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        dt = 1.0 / self.FPS
        self.steps += 1
        self.timer -= dt
        reward = 0.0

        movement, space_held_int, _ = action
        space_held = space_held_int == 1
        space_trigger = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        speed_change = 0.1
        if movement == 1: self.gears[0]['angular_velocity_rps'] += speed_change
        elif movement == 2: self.gears[0]['angular_velocity_rps'] -= speed_change
        if movement == 3: self.gears[1]['angular_velocity_rps'] += speed_change
        elif movement == 4: self.gears[1]['angular_velocity_rps'] -= speed_change

        for gear in self.gears:
            gear['angular_velocity_rps'] = np.clip(gear['angular_velocity_rps'], 1.0, 3.0)

        if space_trigger:
            self._launch_peg() # SFX: Peg launch sound

        self._update_game_logic(dt)
        reward += self._handle_collisions_and_rewards()
        
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.game_over = terminated

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_game_logic(self, dt):
        self.time_since_direction_reversal += dt
        if self.time_since_direction_reversal > 10.0:
            self.time_since_direction_reversal = 0.0
            for gear in self.gears:
                gear['direction'] *= -1
            # SFX: Gear direction change sound

        for gear in self.gears:
            delta_angle = gear['angular_velocity_rps'] * 360 * gear['direction'] * dt
            gear['angle'] = (gear['angle'] + delta_angle) % 360

        pegs_to_remove = [i for i, peg in enumerate(self.pegs) if peg['pos'][1] < 0]
        for i in sorted(pegs_to_remove, reverse=True):
            del self.pegs[i]

        for peg in self.pegs:
            peg['pos'] += peg['vel'] * dt

        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel'] * dt
            p['life'] -= dt

    def _handle_collisions_and_rewards(self):
        reward = 0
        pegs_to_remove = []
        for i, peg in enumerate(self.pegs):
            for gear in self.gears:
                dist_to_center = np.linalg.norm(peg['pos'] - gear['center'])
                if dist_to_center < gear['radius'] + peg['radius']:
                    hit_slot = False
                    for slot_angle_deg in gear['slots']:
                        total_angle_rad = math.radians(gear['angle'] + slot_angle_deg)
                        slot_pos = gear['center'] + np.array([math.cos(total_angle_rad), -math.sin(total_angle_rad)]) * gear['radius']
                        if np.linalg.norm(peg['pos'] - slot_pos) < 12:
                            self.pegs_inserted_count += 1
                            self.score += 10
                            reward += 10
                            if self.pegs_to_insert_total < 15: # Cap total pegs
                                self.pegs_to_insert_total += 1
                            self._create_particles(slot_pos, self.COLOR_SLOT_SUCCESS, 20, 5, 8)
                            # SFX: Success chime
                            hit_slot = True
                            break
                    
                    if hit_slot:
                        pegs_to_remove.append(i)
                    else:
                        reward -= 5
                        self._create_particles(peg['pos'], self.COLOR_PEG, 15, 2, 4)
                        # SFX: Collision clank
                        pegs_to_remove.append(i)
                    break 
        
        for i in sorted(list(set(pegs_to_remove)), reverse=True):
            del self.pegs[i]

        return reward

    def _check_termination(self):
        win = self.pegs_inserted_count >= self.pegs_to_insert_total
        timeout = self.timer <= 0

        if win:
            self.score += 100
            # SFX: Victory fanfare
            return True, 100
        if timeout:
            # SFX: Failure buzzer
            return True, -100
        
        return False, 0

    def _launch_peg(self):
        if not self.pegs:
            peg_pos = np.array([self.WIDTH / 2, self.HEIGHT - 20.0], dtype=float)
            peg_vel = np.array([0.0, -250.0], dtype=float)
            self.pegs.append({'pos': peg_pos, 'vel': peg_vel, 'radius': 6})

    def _create_particles(self, pos, color, count, min_speed, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed * 15
            self.particles.append({
                'pos': pos.copy(), 'vel': vel,
                'life': self.np_random.uniform(0.3, 0.8),
                'color': color, 'radius': self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score, "steps": self.steps, "timer": self.timer,
            "pegs_inserted": self.pegs_inserted_count, "pegs_total": self.pegs_to_insert_total
        }

    def _render_background(self):
        for x, y, size in self.background_stars:
            self.screen.fill((40, 50, 80), (x, y, size, size))

    def _render_game(self):
        for gear in self.gears: self._draw_gear(gear)
        
        launcher_pos = (int(self.WIDTH/2), self.HEIGHT - 15)
        pygame.draw.rect(self.screen, self.COLOR_LAUNCHER, (launcher_pos[0]-15, launcher_pos[1]-5, 30, 10), border_radius=3)
        if not self.pegs:
            pos = (launcher_pos[0], launcher_pos[1] - 15)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, self.COLOR_PEG)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 6, self.COLOR_PEG)

        for peg in self.pegs:
            pos = (int(peg['pos'][0]), int(peg['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(peg['radius']), self.COLOR_PEG)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(peg['radius']), self.COLOR_PEG)

        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius < 1: continue
            alpha = max(0, min(255, int(255 * p['life'])))
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*p['color'], alpha), (radius, radius), radius)
            self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))

    def _draw_gear(self, gear):
        center = (int(gear['center'][0]), int(gear['center'][1]))
        radius = int(gear['radius'])
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, self.COLOR_GEAR)
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, self.COLOR_GEAR_OUTLINE)

        for slot_angle_deg in gear['slots']:
            total_angle_rad = math.radians(gear['angle'] + slot_angle_deg)
            slot_pos = (
                int(center[0] + math.cos(total_angle_rad) * radius),
                int(center[1] - math.sin(total_angle_rad) * radius)
            )
            slot_radius = 10
            pygame.gfxdraw.filled_circle(self.screen, slot_pos[0], slot_pos[1], slot_radius, self.COLOR_SLOT_SUCCESS)
            pygame.gfxdraw.aacircle(self.screen, slot_pos[0], slot_pos[1], slot_radius, self.COLOR_SLOT_SUCCESS)

        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], 8, self.COLOR_GEAR_OUTLINE)

    def _render_ui(self):
        peg_text = f"PEGS: {self.pegs_inserted_count} / {self.pegs_to_insert_total}"
        self._draw_text(peg_text, (20, 15), self.font_small)

        timer_text = f"TIME: {max(0, self.timer):.1f}"
        self._draw_text(timer_text, (self.WIDTH - 150, 15), self.font_small)

        if self.game_over:
            win = self.pegs_inserted_count >= self.pegs_to_insert_total and self.timer > 0
            msg, color = ("SUCCESS", self.COLOR_SLOT_SUCCESS) if win else ("FAILURE", self.COLOR_PEG)
            self._draw_text(msg, (self.WIDTH // 2, self.HEIGHT // 2 + 80), self.font_large, color, centered=True)

    def _draw_text(self, text, pos, font, color=None, centered=False):
        if color is None: color = self.COLOR_UI_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if centered: text_rect.center = pos
        else: text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        self.reset(seed=0)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset(seed=0)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This part of the script is for manual testing and visualization.
    # It will not be used in the evaluation, but is helpful for development.
    # To run this, you'll need to `pip install pygame`.
    
    # Un-comment the line below to run with a display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    try:
        obs, info = env.reset(seed=42)
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Gear Puzzle Environment")
        running = True
        total_reward = 0
        game_clock = pygame.time.Clock()

        print("\n--- Manual Control ---")
        print(GameEnv.user_guide)
        print("--------------------\n")

        while running:
            movement = 0 # none
            space = 0    # released
            shift = 0    # released

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            
            if keys[pygame.K_RIGHT]: movement = 3
            elif keys[pygame.K_LEFT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}")
                print(f"Final Info: {info}")
                print("Resetting environment in 3 seconds...")
                pygame.time.wait(3000)
                obs, info = env.reset(seed=random.randint(0, 1000))
                total_reward = 0
            
            game_clock.tick(env.FPS)
    
    except KeyboardInterrupt:
        print("\nExiting.")
    except pygame.error as e:
        print(f"\nPygame error: {e}")
        print("This example requires a display. If you're running in a headless environment,")
        print("please comment out the line `os.environ.pop(\"SDL_VIDEODRIVER\", None)`")
    finally:
        env.close()