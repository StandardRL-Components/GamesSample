import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:55:37.585689
# Source Brief: brief_01279.md
# Brief Index: 1279
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, min_vel=-1.5, max_vel=1.5, life=20):
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.array([random.uniform(min_vel, max_vel), random.uniform(min_vel, max_vel)], dtype=np.float32)
        self.life = life
        self.color = color
        self.radius = random.uniform(2, 5)

    def update(self):
        self.pos += self.vel
        self.life -= 1
        self.radius -= 0.15
        return self.life > 0 and self.radius > 0

    def draw(self, surface):
        if self.life > 0 and self.radius > 0:
            pygame.draw.circle(surface, self.color, self.pos.astype(int), int(self.radius))

class Orb:
    """Represents a collectible orb."""
    def __init__(self, x, y, radius, color, points, risk, reward_on_collect, penalty_on_fail):
        self.pos = np.array([x, y], dtype=np.float32)
        self.radius = radius
        self.target_radius = radius
        self.current_radius = 0
        self.color = color
        self.points = points
        self.risk = risk
        self.reward_on_collect = reward_on_collect
        self.penalty_on_fail = penalty_on_fail
        self.is_growing = True

    def update(self):
        if self.is_growing and self.current_radius < self.target_radius:
            self.current_radius += (self.target_radius - self.current_radius) * 0.15
            if self.target_radius - self.current_radius < 0.1:
                self.current_radius = self.target_radius
                self.is_growing = False

    def draw(self, surface):
        # Glow effect
        glow_radius = int(self.current_radius * 1.8)
        glow_color = (*self.color, 50)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        surface.blit(temp_surf, (int(self.pos[0] - glow_radius), int(self.pos[1] - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Main circle
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), int(self.current_radius), self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), int(self.current_radius), self.color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Collect orbs to score points while avoiding risky red orbs. "
        "Reach the target score before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to nudge your orb. "
        "Press space to reverse vertical direction."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds
        self.WIN_SCORE = 20

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
        
        # Visuals
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_GREEN = (0, 255, 150)
        self.COLOR_RED = (255, 80, 80)
        self.COLOR_LARGE_RED = (255, 50, 50)
        self.COLOR_UI = (220, 220, 220)
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_feedback = pygame.font.SysFont("Consolas", 20, bold=True)

        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.player_vel = None
        self.player_radius = None
        self.player_trail = None
        self.space_cooldown = None
        self.green_orbs = None
        self.red_orbs = None
        self.large_red_orb = None
        self.orb_shift_timer = None
        self.large_orb_spawn_timer = None
        self.particles = None
        self.reward_feedback_text = None
        self.reward_feedback_color = None
        self.reward_feedback_timer = None
        
        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        angle = self.np_random.uniform(0, 2 * math.pi)
        initial_speed = 3
        self.player_vel = np.array([math.cos(angle) * initial_speed, math.sin(angle) * initial_speed], dtype=np.float32)
        self.player_radius = 12
        self.player_trail = []
        self.space_cooldown = 0
        
        self.green_orbs = []
        self.red_orbs = []
        self._spawn_orbs()
        
        self.large_red_orb = None
        
        self.orb_shift_timer = 5 * self.FPS
        self.large_orb_spawn_timer = 15 * self.FPS
        
        self.particles = []
        self.reward_feedback_text = ""
        self.reward_feedback_color = (0,0,0)
        self.reward_feedback_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        reward = -0.001  # Small penalty for time passing to encourage speed

        self._handle_input(movement, space_held == 1)
        self._update_player()
        self._update_timers()
        self._update_orbs()
        self._update_particles()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        self.steps += 1
        
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += 100
            self._show_reward_feedback(f"+100 WIN! (Score: {self.score})", (100, 255, 100))
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Truncated because it's a time limit, not a failure state
            reward -= 100
            self._show_reward_feedback("-100 TIMEOUT", (255, 100, 100))
        
        self.game_over = terminated or truncated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        nudge_force = 0.2
        if movement == 1: self.player_vel[1] -= nudge_force  # Up
        elif movement == 2: self.player_vel[1] += nudge_force  # Down
        elif movement == 3: self.player_vel[0] -= nudge_force  # Left
        elif movement == 4: self.player_vel[0] += nudge_force  # Right

        max_speed = 6
        speed = np.linalg.norm(self.player_vel)
        if speed > max_speed:
            self.player_vel = self.player_vel * (max_speed / speed)

        if self.space_cooldown > 0:
            self.space_cooldown -= 1
            
        if space_held and self.space_cooldown == 0:
            self.player_vel[1] *= -1
            self.space_cooldown = 15  # 0.5 second cooldown
            # SFX: whoosh
            self._create_particles(self.player_pos, (200, 200, 255), 20, min_vel=-2, max_vel=2)


    def _update_player(self):
        self.player_pos += self.player_vel

        # Wall bounces
        if self.player_pos[0] <= self.player_radius or self.player_pos[0] >= self.WIDTH - self.player_radius:
            self.player_vel[0] *= -1
            self.player_pos[0] = np.clip(self.player_pos[0], self.player_radius, self.WIDTH - self.player_radius)
            # SFX: bounce
            self._create_particles(self.player_pos, self.COLOR_PLAYER, 5)
        if self.player_pos[1] <= self.player_radius or self.player_pos[1] >= self.HEIGHT - self.player_radius:
            self.player_vel[1] *= -1
            self.player_pos[1] = np.clip(self.player_pos[1], self.player_radius, self.HEIGHT - self.player_radius)
            # SFX: bounce
            self._create_particles(self.player_pos, self.COLOR_PLAYER, 5)
        
        # Update trail
        self.player_trail.append(self.player_pos.copy())
        if len(self.player_trail) > 15:
            self.player_trail.pop(0)

    def _update_timers(self):
        self.orb_shift_timer -= 1
        if self.orb_shift_timer <= 0:
            self._spawn_orbs()
            self.orb_shift_timer = 5 * self.FPS
            # SFX: warp/shift sound
        
        self.large_orb_spawn_timer -= 1
        if self.large_orb_spawn_timer <= 0 and self.large_red_orb is None:
            self._spawn_large_red_orb()
            self.large_orb_spawn_timer = 15 * self.FPS
            # SFX: ominous spawn sound
            
        if self.reward_feedback_timer > 0:
            self.reward_feedback_timer -= 1
        else:
            self.reward_feedback_text = ""

    def _update_orbs(self):
        if self.large_red_orb:
            self.large_red_orb.update()
        for orb in self.green_orbs + self.red_orbs:
            orb.update()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _handle_collisions(self):
        reward = 0
        all_orbs = self.green_orbs + self.red_orbs
        if self.large_red_orb:
            all_orbs.append(self.large_red_orb)

        for i in range(len(all_orbs) - 1, -1, -1):
            orb = all_orbs[i]
            dist = np.linalg.norm(self.player_pos - orb.pos)
            if dist < self.player_radius + orb.current_radius:
                # SFX: collect
                self._create_particles(orb.pos, orb.color, 30)
                
                if self.np_random.random() < orb.risk:
                    # SFX: fail/error sound
                    penalty = orb.penalty_on_fail
                    reward += penalty
                    self._show_reward_feedback(f"{penalty} RISK FAILED! Score reset.", (255, 50, 50))
                    self.score = 0
                else:
                    self.score += orb.points
                    reward += orb.reward_on_collect
                    self._show_reward_feedback(f"+{orb.points} pts", orb.color)

                if orb in self.green_orbs: self.green_orbs.remove(orb)
                elif orb in self.red_orbs: self.red_orbs.remove(orb)
                elif orb == self.large_red_orb: self.large_red_orb = None
        return reward

    def _spawn_orbs(self):
        self.green_orbs.clear()
        self.red_orbs.clear()
        for _ in range(8):
            pos = self.np_random.uniform([50, 50], [self.WIDTH - 50, self.HEIGHT - 50])
            self.green_orbs.append(Orb(pos[0], pos[1], 7, self.COLOR_GREEN, 1, 0, 0.1, 0))
        for _ in range(4):
            pos = self.np_random.uniform([50, 50], [self.WIDTH - 50, self.HEIGHT - 50])
            self.red_orbs.append(Orb(pos[0], pos[1], 8, self.COLOR_RED, 3, 0.2, 0.3, -10))

    def _spawn_large_red_orb(self):
        pos = self.np_random.uniform([100, 100], [self.WIDTH - 100, self.HEIGHT - 100])
        self.large_red_orb = Orb(pos[0], pos[1], 18, self.COLOR_LARGE_RED, 5, 0.5, 0.5, -50)

    def _create_particles(self, pos, color, count, min_vel=-1, max_vel=1):
        for _ in range(count):
            self.particles.append(Particle(pos[0], pos[1], color, min_vel, max_vel))
            
    def _show_reward_feedback(self, text, color):
        self.reward_feedback_text = text
        self.reward_feedback_color = color
        self.reward_feedback_timer = self.FPS * 1.5 # 1.5 seconds

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Trail
        for i, pos in enumerate(self.player_trail):
            alpha = int(255 * (i / len(self.player_trail)))
            radius = int(self.player_radius * 0.5 * (i / len(self.player_trail)))
            if radius > 0:
                pygame.draw.circle(self.screen, (*self.COLOR_PLAYER, alpha), pos.astype(int), radius)

        # Particles
        for p in self.particles:
            p.draw(self.screen)
        
        # Orbs
        for orb in self.green_orbs + self.red_orbs:
            orb.draw(self.screen)
        if self.large_red_orb:
            self.large_red_orb.draw(self.screen)

        # Player
        self._draw_glowing_circle(self.screen, self.player_pos, self.player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_color = self.COLOR_UI if time_left > 10 else (255, 100, 100)
        timer_text = self.font_main.render(f"TIME: {time_left:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        # Reward feedback
        if self.reward_feedback_timer > 0:
            alpha = min(255, int(255 * (self.reward_feedback_timer / (self.FPS * 0.5))))
            color = (*self.reward_feedback_color, alpha)
            feedback_surf = self.font_feedback.render(self.reward_feedback_text, True, color)
            feedback_surf.set_alpha(alpha)
            pos = self.player_pos - np.array([feedback_surf.get_width() / 2, 50])
            self.screen.blit(feedback_surf, pos.astype(int))

    def _draw_glowing_circle(self, surface, pos, radius, color):
        # Glow
        glow_radius = int(radius * 2.5)
        glow_color = (*color, 30)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        surface.blit(temp_surf, (int(pos[0] - glow_radius), int(pos[1] - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Main circle
        pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), int(radius), color)
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius), color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # Example usage: Play the game manually
    # The main loop needs a visible display, so we unset the dummy driver
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Orb Collector")
    clock = pygame.time.Clock()

    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("Arrows: Move")
    print("Space: Reverse vertical direction")
    print("R: Reset")
    print("Q: Quit")
    
    running = True
    while running:
        # Action defaults
        movement = 0
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    done = False

        if not done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = np.array([movement, space, shift])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()