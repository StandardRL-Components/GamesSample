import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:53:30.967812
# Source Brief: brief_03071.md
# Brief Index: 3071
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent must sort items on conveyor belts.
    The agent controls the speed of three belts to arrange items ('A' through 'J')
    in alphabetical order as they exit the belts.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Adjust the speed of three conveyor belts to sort items alphabetically as they pass. "
        "Correctly sorted items score points, while mistakes reset an item's position."
    )
    user_guide = (
        "Use ↑ to select the top belt, ↓ for the middle belt, and ← for the bottom belt. "
        "Hold space to increase the selected belt's speed, or shift to decrease it."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 2700  # 90 seconds at 30 FPS
    WIN_SCORE = 700
    FPS = 30

    COLOR_BG = (25, 35, 45)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_BELT = (60, 70, 80)
    COLOR_BELT_SLOW = (60, 180, 100)
    COLOR_BELT_FAST = (220, 80, 80)
    COLOR_SELECT_GLOW = (255, 255, 100)

    BELT_Y_POSITIONS = [100, 200, 300]
    BELT_HEIGHT = 50
    ITEM_RADIUS = 18
    BASE_ITEM_SPEED = 6.0

    ITEM_CHARS = "ABCDEFGHIJ"
    ITEM_COLORS = {
        'A': (255, 87, 87), 'B': (255, 170, 87), 'C': (255, 255, 87),
        'D': (170, 255, 87), 'E': (87, 255, 87), 'F': (87, 255, 170),
        'G': (87, 255, 255), 'H': (87, 170, 255), 'I': (170, 87, 255),
        'J': (255, 87, 255)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_item = pygame.font.SysFont("Arial", 20, bold=True)
        
        self.steps = None
        self.score = None
        self.game_over = None
        self.belt_speeds = None
        self.items = None
        self.particles = None
        self.last_sorted_char = None
        self.selected_belt_idx = None
        self.next_item_spawn_step = None
        self.last_milestone_score = None
        self.belt_anim_offsets = None

        # self.reset() # reset is called by the wrapper/runner
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.belt_speeds = [self.np_random.uniform(0.2, 1.0) for _ in range(3)]
        self.items = []
        self.particles = []
        self.last_sorted_char = ['@'] * 3  # '@' is before 'A'
        self.selected_belt_idx = -1
        self.next_item_spawn_step = 30 # Spawn first item after 1 sec
        self.last_milestone_score = 0
        self.belt_anim_offsets = [0.0, 0.0, 0.0]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        self._handle_action(action)
        self._update_belts()
        self._spawn_items()
        reward += self._update_items()
        self._update_particles()
        
        if self.score // 100 > self.last_milestone_score // 100:
            milestones_gained = (self.score // 100) - (self.last_milestone_score // 100)
            reward += 10 * milestones_gained
            self.last_milestone_score = self.score
            # SFX: Milestone reached sound

        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
            # SFX: Win sound
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Time limit reached is termination
            self.game_over = True
            # SFX: Lose sound
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement selects the belt (1=up, 2=down, 3=left)
        if movement == 1: self.selected_belt_idx = 0
        elif movement == 2: self.selected_belt_idx = 1
        elif movement == 3: self.selected_belt_idx = 2
        # movement 0 (none) and 4 (right) do not change selection

        if self.selected_belt_idx != -1:
            speed_change = 0.0
            if space_held and not shift_held:
                speed_change = 0.01  # Increase speed
                # SFX: Speed up click
            elif shift_held and not space_held:
                speed_change = -0.01 # Decrease speed
                # SFX: Speed down click

            if speed_change != 0.0:
                current_speed = self.belt_speeds[self.selected_belt_idx]
                new_speed = np.clip(current_speed + speed_change, 0.2, 1.0)
                self.belt_speeds[self.selected_belt_idx] = new_speed

    def _update_belts(self):
        for i in range(3):
            self.belt_anim_offsets[i] = (self.belt_anim_offsets[i] + self.belt_speeds[i] * self.BASE_ITEM_SPEED * 0.5) % 40

    def _spawn_items(self):
        if self.steps >= self.next_item_spawn_step:
            belt_idx = self.np_random.integers(0, 3)
            char = self.np_random.choice(list(self.ITEM_CHARS))
            
            can_spawn = True
            for item in self.items:
                if item['belt_idx'] == belt_idx and item['pos'][0] < self.ITEM_RADIUS * 2.5:
                    can_spawn = False
                    break
            
            if can_spawn:
                self.items.append({
                    'char': char,
                    'pos': np.array([float(self.ITEM_RADIUS), float(self.BELT_Y_POSITIONS[belt_idx])]),
                    'belt_idx': belt_idx
                })
                self.next_item_spawn_step = self.steps + self.np_random.integers(75, 105) # 2.5-3.5 seconds
                # SFX: Item spawn sound

    def _update_items(self):
        step_reward = 0
        items_to_remove = []
        for i, item in enumerate(self.items):
            item['pos'][0] += self.belt_speeds[item['belt_idx']] * self.BASE_ITEM_SPEED

            if item['pos'][0] > self.WIDTH + self.ITEM_RADIUS:
                belt_idx = item['belt_idx']
                if item['char'] > self.last_sorted_char[belt_idx]:
                    self.score += 10
                    step_reward += 0.1
                    self.last_sorted_char[belt_idx] = item['char']
                    self._create_particles(item['pos'], self.ITEM_COLORS[item['char']])
                    items_to_remove.append(i)
                    # SFX: Correct sort chime
                else:
                    item['pos'][0] = float(self.ITEM_RADIUS)
                    # SFX: Incorrect sort buzz
        
        for i in sorted(items_to_remove, reverse=True):
            del self.items[i]
            
        return step_reward

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'color': color})
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        for i in range(3):
            y_pos = self.BELT_Y_POSITIONS[i] - self.BELT_HEIGHT // 2
            belt_rect = pygame.Rect(0, y_pos, self.WIDTH, self.BELT_HEIGHT)
            
            speed_ratio = (self.belt_speeds[i] - 0.2) / 0.8
            belt_color = [int(np.interp(speed_ratio, [0, 1], [self.COLOR_BELT_SLOW[c], self.COLOR_BELT_FAST[c]])) for c in range(3)]
            pygame.draw.rect(self.screen, belt_color, belt_rect, border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_BELT, belt_rect.inflate(-6,-6), border_radius=5)

            for offset in range(0, self.WIDTH, 40):
                dot_x = int((offset + self.belt_anim_offsets[i]) % self.WIDTH)
                pygame.draw.circle(self.screen, (0,0,0,50), (dot_x, int(y_pos + self.BELT_HEIGHT/2)), 3)

            if self.selected_belt_idx == i:
                self._render_glow(belt_rect)

        for p in self.particles:
            alpha = int(np.interp(p['lifespan'], [0, 30], [0, 255]))
            alpha = max(0, min(255, alpha))
            color = p['color']
            
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, (color[0], color[1], color[2], alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, (color[0], color[1], color[2], alpha))

        for item in self.items:
            pos = (int(item['pos'][0]), int(item['pos'][1]))
            color = self.ITEM_COLORS[item['char']]
            
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ITEM_RADIUS + 2, (color[0], color[1], color[2], 60))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ITEM_RADIUS, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ITEM_RADIUS, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ITEM_RADIUS - 3, (30,30,30))
            
            text_surf = self.font_item.render(item['char'], True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=pos)
            self.screen.blit(text_surf, text_rect)

    def _render_glow(self, rect):
        for i in range(4, 0, -1):
            alpha = 60 - i * 10
            glow_rect = rect.inflate(i * 3, i * 3)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.COLOR_SELECT_GLOW, alpha), glow_surf.get_rect(), border_radius=8)
            self.screen.blit(glow_surf, glow_rect.topleft)

    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"TIME: {time_left:.1f}"
        time_surf = self.font_main.render(time_text, True, self.COLOR_UI_TEXT)
        time_rect = time_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_surf, time_rect)

        for i in range(3):
            y_pos = self.BELT_Y_POSITIONS[i] - self.BELT_HEIGHT // 2
            speed_pct = int(self.belt_speeds[i] * 100)
            speed_text = f"{speed_pct}%"
            speed_surf = self.font_main.render(speed_text, True, self.COLOR_UI_TEXT)
            speed_rect = speed_surf.get_rect(center=(self.WIDTH // 2, y_pos - 20))
            self.screen.blit(speed_surf, speed_rect)
            
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block is for manual play and debugging.
    # It will not be executed by the test harness.
    # We can use a different display driver for manual play.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Conveyor Belt Sort")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    movement = 0
    space_held = 0
    shift_held = 0
    
    print("\n--- Manual Control ---")
    print("W/S/A: Select Belts 1/2/3 (Corresponds to agent actions 1, 2, 3)")
    print("UP ARROW: Increase speed (Spacebar action)")
    print("DOWN ARROW: Decrease speed (Shift action)")
    print("R: Reset environment")
    print("Q: Quit")
    print("---------------------\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print(f"Environment Reset. Initial Info: {info}")
                
                if event.key == pygame.K_w: movement = 1
                if event.key == pygame.K_s: movement = 2
                if event.key == pygame.K_a: movement = 3
                if event.key == pygame.K_UP: space_held = 1
                if event.key == pygame.K_DOWN: shift_held = 1
                    
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_w, pygame.K_s, pygame.K_a]: movement = 0
                if event.key == pygame.K_UP: space_held = 0
                if event.key == pygame.K_DOWN: shift_held = 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Final Info: {info}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            # Reset action states for the new episode
            movement = 0
            space_held = 0
            shift_held = 0

        clock.tick(GameEnv.FPS)

    env.close()